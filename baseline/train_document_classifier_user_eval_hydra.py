from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import train_document_classifier as base
from common import REL_IDS, REL_ID_TO_NAME


def assign_distance_bucket(distance, boundaries):
    distance = float(distance)
    for idx, boundary in enumerate(boundaries):
        if distance < boundary:
            return idx
    return len(boundaries)


def f1_from_sets(pred_set, gold_set):
    correct = pred_set & gold_set
    precision = len(correct) / (len(pred_set) + 1e-5)
    recall = len(correct) / (len(gold_set) + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return precision * 100, recall * 100, f1 * 100, len(correct)


def evaluate_like_user_code(model, loader, data, args, device, split_name):
    tag = split_name.split("-")[0]
    model.eval()
    pred_facts = set()
    gold_facts = set()
    fact_dist = {}
    fact_bucket = {}
    details = []
    total_docs = 0
    total_pairs = 0
    tn = 0.0
    fp_raw = 0.0
    boundaries = [8, 32, 64, 128]

    print(f"[Eval/UserStyle] split={split_name} docs={len(data)} batch_size={args.eval_batch_size} threshold={args.threshold}")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {split_name}", dynamic_ncols=True, leave=False):
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            docs = batch["docs"]
            logit_list = model(encoded, docs)
            for doc, logits in zip(docs, logit_list):
                total_docs += 1
                total_pairs += len(doc["pair_indices"])
                if logits.numel() == 0:
                    continue

                scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                golds = np.asarray(doc["labels"], dtype=np.float32)
                rel_pred = scores >= float(args.threshold)
                rel_gold = golds == 1

                tn += ((~rel_pred) & (~rel_gold)).astype(np.float32).sum()
                fp_raw += (rel_pred & (~rel_gold)).astype(np.float32).sum()

                for row_idx, meta in enumerate(doc["pair_meta"]):
                    row_rels = []
                    dist = int(meta.get("dist", 0))
                    bucket = assign_distance_bucket(meta.get("ent_dis", 0), boundaries)
                    for rel_col, rel_id in enumerate(REL_IDS):
                        rel_name = REL_ID_TO_NAME[rel_id]
                        fact = (meta["title"], rel_name, int(meta["h"]), int(meta["t"]))
                        fact_dist[fact] = dist
                        fact_bucket[fact] = bucket
                        if rel_pred[row_idx, rel_col]:
                            pred_facts.add(fact)
                            row_rels.append({"r": rel_id, "relation": rel_name, "score": float(scores[row_idx, rel_col])})
                        if rel_gold[row_idx, rel_col]:
                            gold_facts.add(fact)
                    if row_rels:
                        details.append(
                            {
                                "title": meta["title"],
                                "h": int(meta["h"]),
                                "t": int(meta["t"]),
                                "head": meta["head"],
                                "tail": meta["tail"],
                                "relations": row_rels,
                            }
                        )

    p, r, f1, correct_count = f1_from_sets(pred_facts, gold_facts)
    fer = tn / (tn + fp_raw + 1e-5) * 100

    pred_intra = {fact for fact in pred_facts if fact_dist.get(fact) == 0}
    gold_intra = {fact for fact in gold_facts if fact_dist.get(fact) == 0}
    _, _, f1_intra, _ = f1_from_sets(pred_intra, gold_intra)

    pred_inter = {fact for fact in pred_facts if fact_dist.get(fact) == 1}
    gold_inter = {fact for fact in gold_facts if fact_dist.get(fact) == 1}
    _, _, f1_inter, _ = f1_from_sets(pred_inter, gold_inter)

    bucket_f1 = []
    for bucket in range(len(boundaries) + 1):
        pred_bucket = {fact for fact in pred_facts if fact_bucket.get(fact) == bucket}
        gold_bucket = {fact for fact in gold_facts if fact_bucket.get(fact) == bucket}
        _, _, bucket_score, _ = f1_from_sets(pred_bucket, gold_bucket)
        bucket_f1.append(bucket_score)

    metrics = {
        "p": p,
        "r": r,
        "fer": fer,
        "f1": f1,
        "f1_intra": f1_intra,
        "f1_inter": f1_inter,
        "f1_1": bucket_f1[0],
        "f1_2": bucket_f1[1],
        "f1_3": bucket_f1[2],
        "f1_4": bucket_f1[3],
        "f1_5": bucket_f1[4],
        "gold": len(gold_facts),
        "pred": len(pred_facts),
        "correct": correct_count,
    }
    print(
        f"[Eval/UserStyle] split={split_name} docs_seen={total_docs} pairs_seen={total_pairs} "
        f"p={p:.2f} r={r:.2f} fer={fer:.2f} f1={f1:.2f} "
        f"gold={len(gold_facts)} pred={len(pred_facts)} correct={correct_count}"
    )
    return metrics, details


def format_metric_output_like_user_code(tag, metrics):
    return {
        f"{tag}_p": metrics["p"],
        f"{tag}_r": metrics["r"],
        f"{tag}_fer": metrics["fer"],
        f"{tag}_f1": metrics["f1"],
        f"{tag}_f1_intra": metrics["f1_intra"],
        f"{tag}_f1_inter": metrics["f1_inter"],
        f"{tag}_f1_1": metrics["f1_1"],
        f"{tag}_f1_2": metrics["f1_2"],
        f"{tag}_f1_3": metrics["f1_3"],
        f"{tag}_f1_4": metrics["f1_4"],
        f"{tag}_f1_5": metrics["f1_5"],
    }


@hydra.main(config_path="configs", config_name="doc_roberta", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config_dir = Path(__file__).resolve().parent / "configs"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = base.namespace_from_hydra_config(cfg_dict, config_dir)

    base.run_eval = evaluate_like_user_code
    base.format_metric_output = format_metric_output_like_user_code
    base.train(args)


if __name__ == "__main__":
    main()
