import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import (
    REL_IDS,
    REL_ID_TO_INDEX,
    REL_ID_TO_NAME,
    evaluate_triples,
    load_json,
    print_per_relation_metrics,
    print_scope_metrics,
    rel_id,
    save_json,
    save_metric_tables,
)
from train_pair_classifier import resolve_model_name, resolve_path


SPECIAL_TOKENS = ["[SENT]"]


def format_elapsed(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(args):
    if args.cpu or str(args.device).lower() == "cpu":
        return torch.device("cpu")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: requested {args.device}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(args.device)


def load_tokenizer(model_name):
    kwargs = {"use_fast": True}
    if "roberta" in str(model_name).lower():
        kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


def mention_text(mention):
    name = mention.get("name", "")
    if isinstance(name, list):
        return " ".join(str(x) for x in name)
    return str(name)


def entity_name(doc, ent_idx):
    return mention_text(doc["vertexSet"][ent_idx][0])


def all_ordered_entity_pairs(doc):
    n = len(doc.get("vertexSet", []))
    for h in range(n):
        for t in range(n):
            if h != t:
                yield h, t


def gold_relation_map(doc):
    pair_to_rels = defaultdict(set)
    for label in doc.get("labels", []):
        pair_to_rels[(int(label["h"]), int(label["t"]))].add(rel_id(label))
    return pair_to_rels


def label_vector(rel_ids):
    vec = [0.0] * len(REL_IDS)
    for r in rel_ids:
        if r in REL_ID_TO_INDEX:
            vec[REL_ID_TO_INDEX[r]] = 1.0
    return vec


def count_gold_relations(data):
    counts = Counter()
    total = 0
    for doc in data:
        for label in doc.get("labels", []):
            rid = rel_id(label)
            counts[rid] += 1
            total += 1
    return total, counts


def print_relation_counts(prefix, counts):
    pieces = []
    for rid in REL_IDS:
        value = counts.get(rid, 0)
        if value:
            pieces.append(f"{REL_ID_TO_NAME.get(rid, rid)}={value}")
    print(f"{prefix}: " + (", ".join(pieces) if pieces else "none"))


class DocREDDocumentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, split, training, neg_ratio, seed):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.training = training
        self.neg_ratio = neg_ratio
        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx)
        return encode_document(
            self.data[idx],
            self.tokenizer,
            self.max_length,
            training=self.training,
            neg_ratio=self.neg_ratio,
            rng=rng,
        )


def flatten_document_words(doc):
    words = []
    word_map = {}
    global_word_map = {}
    global_cursor = 0
    for sent_id, sent in enumerate(doc.get("sents", [])):
        for word_id, word in enumerate(sent):
            flat_idx = len(words)
            word_map[(sent_id, word_id)] = flat_idx
            global_word_map[global_cursor] = flat_idx
            words.append(str(word))
            global_cursor += 1
        words.append("[SENT]")
    return words, word_map, global_word_map


def mention_flat_word_indices(mention, word_map, global_word_map=None):
    if global_word_map is not None and "global_pos" in mention:
        start, end = int(mention["global_pos"][0]), int(mention["global_pos"][1])
        indices = [global_word_map[i] for i in range(start, end) if i in global_word_map]
        if indices:
            return indices

    sent_id = int(mention["sent_id"])
    start, end = int(mention["pos"][0]), int(mention["pos"][1])
    indices = []
    for word_id in range(start, end):
        value = word_map.get((sent_id, word_id))
        if value is not None:
            indices.append(value)
    return indices


def word_piece_positions(encoding):
    word_to_pieces = defaultdict(list)
    for piece_idx, word_idx in enumerate(encoding.word_ids()):
        if word_idx is not None:
            word_to_pieces[int(word_idx)].append(piece_idx)
    return word_to_pieces


def visible_entity_piece_indices(doc, word_map, global_word_map, word_to_pieces):
    entity_pieces = []
    for entity in doc.get("vertexSet", []):
        pieces = []
        for mention in entity:
            for word_idx in mention_flat_word_indices(mention, word_map, global_word_map):
                pieces.extend(word_to_pieces.get(word_idx, []))
        pieces = sorted(set(pieces))
        entity_pieces.append(pieces)
    return entity_pieces



def first_mention_span(doc, ent_idx):
    mentions = doc.get("vertexSet", [])[ent_idx]
    if not mentions:
        return None
    mention = mentions[0]
    if "global_pos" in mention:
        return int(mention["global_pos"][0]), int(mention["global_pos"][1])
    sent_offset = sum(len(sent) for sent in doc.get("sents", [])[: int(mention["sent_id"])])
    return sent_offset + int(mention["pos"][0]), sent_offset + int(mention["pos"][1])


def entity_distance(doc, h, t):
    head_span = first_mention_span(doc, h)
    tail_span = first_mention_span(doc, t)
    if head_span is None or tail_span is None:
        return 0
    if head_span[1] < tail_span[0]:
        return tail_span[0] - head_span[1]
    if head_span[0] > tail_span[1]:
        return head_span[0] - tail_span[1]
    return 0


def pair_cross_sentence(doc, h, t, rels):
    for label in doc.get("labels", []):
        if int(label.get("h", -1)) == int(h) and int(label.get("t", -1)) == int(t):
            dist = str(label.get("dist", "")).upper()
            if dist == "CROSS":
                return 1
            if dist == "NON-CROSS":
                return 0

    h_sents = {int(m["sent_id"]) for m in doc.get("vertexSet", [])[h]}
    t_sents = {int(m["sent_id"]) for m in doc.get("vertexSet", [])[t]}
    return 0 if h_sents & t_sents else 1
def encode_document(doc, tokenizer, max_length, training, neg_ratio, rng):
    words, word_map, global_word_map = flatten_document_words(doc)
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    word_to_pieces = word_piece_positions(encoding)
    entity_pieces = visible_entity_piece_indices(doc, word_map, global_word_map, word_to_pieces)
    gold = gold_relation_map(doc)

    positive_pairs = []
    negative_pairs = []
    for h, t in all_ordered_entity_pairs(doc):
        if not entity_pieces[h] or not entity_pieces[t]:
            continue
        rels = gold.get((h, t), set())
        pair = (h, t, rels)
        if rels:
            positive_pairs.append(pair)
        else:
            negative_pairs.append(pair)

    if training and negative_pairs and neg_ratio >= 0:
        keep = min(len(negative_pairs), int(max(1, len(positive_pairs)) * neg_ratio))
        negative_pairs = rng.sample(negative_pairs, keep)
    pairs = positive_pairs + negative_pairs
    rng.shuffle(pairs)

    pair_indices = []
    label_rows = []
    pair_meta = []
    for h, t, rels in pairs:
        pair_indices.append([h, t])
        label_rows.append(label_vector(rels))
        pair_meta.append(
            {
                "title": doc.get("title", ""),
                "h": int(h),
                "t": int(t),
                "head": entity_name(doc, h),
                "tail": entity_name(doc, t),
                "dist": pair_cross_sentence(doc, h, t, rels),
                "ent_dis": entity_distance(doc, h, t),
            }
        )

    return {
        "title": doc.get("title", ""),
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "token_type_ids": encoding.get("token_type_ids"),
        "entity_pieces": entity_pieces,
        "pair_indices": pair_indices,
        "labels": label_rows,
        "pair_meta": pair_meta,
        "num_entities": len(entity_pieces),
        "num_positive_pairs": len(positive_pairs),
        "num_negative_pairs": len(negative_pairs),
        "num_visible_pairs": len(pairs),
    }


class DocumentCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, docs):
        encoded = []
        for doc in docs:
            item = {"input_ids": doc["input_ids"], "attention_mask": doc["attention_mask"]}
            if doc.get("token_type_ids") is not None:
                item["token_type_ids"] = doc["token_type_ids"]
            encoded.append(item)
        padded = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        return {"encoded": padded, "docs": docs}


class AdaptiveThresholdLoss(nn.Module):
    """ATLOP-style adaptive threshold loss for multi-label relation logits.

    Input logits have shape [num_pairs, 1 + num_relations]. Column 0 is the
    learnable threshold/NA class; columns 1: are positive relation classes.
    Labels keep the existing baseline shape [num_pairs, num_relations].
    """

    def forward(self, logits, labels):
        full_labels = torch.zeros_like(logits, dtype=torch.float)
        full_labels[:, 1:] = labels.float()
        th_label = torch.zeros_like(logits, dtype=torch.float)
        th_label[:, 0] = 1.0

        p_mask = full_labels + th_label
        n_mask = 1.0 - full_labels

        logit_pos = logits - (1.0 - p_mask) * 1e30
        loss_pos = -(F.log_softmax(logit_pos, dim=-1) * full_labels).sum(dim=1)

        logit_neg = logits - (1.0 - n_mask) * 1e30
        loss_neg = -(F.log_softmax(logit_neg, dim=-1) * th_label).sum(dim=1)
        return (loss_pos + loss_neg).mean()


class DocumentREModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def resize_token_embeddings(self, size):
        self.encoder.resize_token_embeddings(size)

    def forward(self, encoded, docs):
        kwargs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded:
            kwargs["token_type_ids"] = encoded["token_type_ids"]
        hidden = self.encoder(**kwargs).last_hidden_state
        outputs = []
        for batch_idx, doc in enumerate(docs):
            entity_reps = []
            for pieces in doc["entity_pieces"]:
                if pieces:
                    idx = torch.tensor(pieces, dtype=torch.long, device=hidden.device)
                    entity_reps.append(hidden[batch_idx, idx].mean(dim=0))
                else:
                    entity_reps.append(torch.zeros(hidden.size(-1), dtype=hidden.dtype, device=hidden.device))
            if not doc["pair_indices"]:
                outputs.append(hidden.new_zeros((0, len(REL_IDS))))
                continue
            entity_reps = torch.stack(entity_reps, dim=0)
            pair_idx = torch.tensor(doc["pair_indices"], dtype=torch.long, device=hidden.device)
            head = entity_reps[pair_idx[:, 0]]
            tail = entity_reps[pair_idx[:, 1]]
            pair_rep = torch.cat([head, tail, torch.abs(head - tail), head * tail], dim=-1)
            outputs.append(self.classifier(self.dropout(pair_rep)))
        return outputs


def make_loader(data, tokenizer, args, split, training):
    dataset = DocREDDocumentDataset(
        data,
        tokenizer,
        max_length=args.max_length,
        split=split,
        training=training,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    batch_size = args.batch_size if training else args.eval_batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=DocumentCollator(tokenizer),
        num_workers=0,
    )


def collect_batch_labels(docs, device):
    labels = []
    for doc in docs:
        if doc["labels"]:
            labels.append(torch.tensor(doc["labels"], dtype=torch.float, device=device))
    if not labels:
        return None
    return torch.cat(labels, dim=0)


def collect_batch_logits(logit_list):
    non_empty = [x for x in logit_list if x.numel() > 0]
    if not non_empty:
        return None
    return torch.cat(non_empty, dim=0)


def run_eval(model, loader, data, args, device, split_name):
    model.eval()
    pred_triples = []
    details = []
    total_docs = 0
    total_pairs = 0
    print(f"[Eval] split={split_name} docs={len(data)} batch_size={args.eval_batch_size} threshold={args.threshold}")
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
                if args.loss_type == "adaptive_threshold":
                    rows = logits.detach().cpu().numpy()
                    for meta, row in zip(doc["pair_meta"], rows):
                        rels = []
                        threshold_logit = float(row[0])
                        for idx, score in enumerate(row[1:]):
                            if float(score) > threshold_logit:
                                rid = REL_IDS[idx]
                                pred_triples.append((meta["title"], int(meta["h"]), int(meta["t"]), rid))
                                rels.append({
                                    "r": rid,
                                    "relation": REL_ID_TO_NAME[rid],
                                    "score": float(score),
                                    "threshold": threshold_logit,
                                    "margin": float(score) - threshold_logit,
                                })
                        if rels:
                            details.append(
                                {
                                    "title": meta["title"],
                                    "h": int(meta["h"]),
                                    "t": int(meta["t"]),
                                    "head": meta["head"],
                                    "tail": meta["tail"],
                                    "relations": rels,
                                }
                            )
                else:
                    scores = torch.sigmoid(logits).detach().cpu().numpy()
                    for meta, row in zip(doc["pair_meta"], scores):
                        rels = []
                        for idx, score in enumerate(row):
                            if float(score) >= args.threshold:
                                rid = REL_IDS[idx]
                                pred_triples.append((meta["title"], int(meta["h"]), int(meta["t"]), rid))
                                rels.append({"r": rid, "relation": REL_ID_TO_NAME[rid], "score": float(score)})
                        if rels:
                            details.append(
                                {
                                    "title": meta["title"],
                                    "h": int(meta["h"]),
                                    "t": int(meta["t"]),
                                    "head": meta["head"],
                                    "tail": meta["tail"],
                                    "relations": rels,
                                }
                            )
    metrics = evaluate_triples(data, pred_triples)
    print(
        f"[Eval] split={split_name} docs_seen={total_docs} pairs_seen={total_pairs} "
        f"p={metrics['p']:.2f} r={metrics['r']:.2f} f1={metrics['f1']:.2f} "
        f"gold={metrics['gold']} pred={metrics['pred']} correct={metrics['correct']}"
    )
    print_scope_metrics(split_name, metrics)
    if split_name.endswith("final"):
        print_per_relation_metrics(split_name, metrics)
    return metrics, details



def format_metric_output(tag, metrics):
    return {
        f"{tag}_p": metrics["p"],
        f"{tag}_r": metrics["r"],
        f"{tag}_f1": metrics["f1"],
        f"{tag}_f1_intra": metrics.get("f1_intra", 0.0),
        f"{tag}_f1_inter": metrics.get("f1_inter", 0.0),
        f"{tag}_gold": metrics["gold"],
        f"{tag}_pred": metrics["pred"],
        f"{tag}_correct": metrics["correct"],
    }


def print_metric_output(tag, metrics):
    print(format_metric_output(tag, metrics))

def print_run_header(args):
    print("=" * 88)
    print("Maritime RE Document-Batch Baseline")
    print(f"pid={os.getpid()}")
    print(f"config={args.config}")
    print(f"model_name={args.model_name}")
    print(f"train_path={args.train_path}")
    print(f"dev_path={args.dev_path}")
    print(f"test_path={args.test_path}")
    print(f"output_dir={args.output_dir}")
    print(
        f"epochs={args.epochs} train_batch_size={args.batch_size} eval_batch_size={args.eval_batch_size} "
        f"lr={args.lr} max_length={args.max_length} threshold={args.threshold} neg_ratio={args.neg_ratio} "
        f"loss_type={args.loss_type} eval_test_each_epoch={args.eval_test_each_epoch}"
    )
    print("=" * 88)


def train(args):
    job_start = time.time()
    set_seed(args.seed)
    print_run_header(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "effective_args.json", vars(args), pretty=True)

    print("[Data] loading JSON files...")
    t0 = time.time()
    train_data = load_json(args.train_path)
    dev_data = load_json(args.dev_path)
    test_data = load_json(args.test_path)
    print(
        f"[Data] loaded in {format_elapsed(time.time() - t0)} | "
        f"train_docs={len(train_data)} dev_docs={len(dev_data)} test_docs={len(test_data)}"
    )
    for split_name, split_data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        total_labels, rel_counts = count_gold_relations(split_data)
        print(f"[Data] {split_name}: docs={len(split_data)} gold_labels={total_labels}")
        print_relation_counts(f"[Data] {split_name} relation_counts", rel_counts)

    print("[Tokenizer] loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)
    print(f"[Tokenizer] vocab_size={len(tokenizer)}")

    print("[Loader] building document-level dataloaders...")
    train_loader = make_loader(train_data, tokenizer, args, split="train", training=True)
    dev_loader = make_loader(dev_data, tokenizer, args, split="dev", training=False)
    test_loader = make_loader(test_data, tokenizer, args, split="test", training=False)
    print(
        f"[Loader] train_steps_per_epoch={len(train_loader)} dev_steps={len(dev_loader)} test_steps={len(test_loader)}"
    )

    device = choose_device(args)
    print(f"[Model] loading model on device={device}...")
    output_labels = len(REL_IDS) + (1 if args.loss_type == "adaptive_threshold" else 0)
    model = DocumentREModel(args.model_name, output_labels, dropout=args.dropout)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"[Model] hidden_size={model.encoder.config.hidden_size} relation_labels={len(REL_IDS)} output_labels={output_labels}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = AdaptiveThresholdLoss() if args.loss_type == "adaptive_threshold" else nn.BCEWithLogitsLoss()
    print(
        f"[Train] total_steps={total_steps} steps_per_epoch={len(train_loader)} "
        f"warmup_steps={int(total_steps * args.warmup_ratio)} log_steps={args.log_steps}"
    )

    best_dev = -1.0
    best_path = output_dir / "best_model.pt"
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_pairs = 0
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
            encoded = {k: v.to(device) for k, v in batch["encoded"].items()}
            docs = batch["docs"]
            optimizer.zero_grad()
            logit_list = model(encoded, docs)
            logits = collect_batch_logits(logit_list)
            labels = collect_batch_labels(docs, device)
            if logits is None or labels is None:
                continue
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.item())
            total_loss += loss_value
            batch_pairs = int(labels.size(0))
            total_pairs += batch_pairs
            pbar.set_postfix(loss=f"{loss_value:.4f}", pairs=batch_pairs)
            if args.log_steps > 0 and global_step % args.log_steps == 0:
                print(
                    f"[Train] epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"global_step={global_step}/{total_steps} loss={loss_value:.4f} batch_pairs={batch_pairs}"
                )

        dev_metrics, _ = run_eval(model, dev_loader, dev_data, args, device, split_name="dev")
        print_metric_output("dev", dev_metrics)
        if args.eval_test_each_epoch:
            test_metrics_epoch, _ = run_eval(model, test_loader, test_data, args, device, split_name="test")
            print_metric_output("test", test_metrics_epoch)
        avg_loss = total_loss / max(1, len(train_loader))
        print(
            f"[Epoch] epoch={epoch}/{args.epochs} loss={avg_loss:.4f} train_pairs={total_pairs} "
            f"dev_p={dev_metrics['p']:.2f} dev_r={dev_metrics['r']:.2f} dev_f1={dev_metrics['f1']:.2f} "
            f"elapsed={format_elapsed(time.time() - epoch_start)}"
        )
        if dev_metrics["f1"] > best_dev:
            best_dev = dev_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"[Checkpoint] new best dev_f1={best_dev:.2f}; saved to {best_path}")

    if best_path.exists():
        print(f"[Checkpoint] loading best checkpoint from {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))

    dev_metrics, dev_preds = run_eval(model, dev_loader, dev_data, args, device, split_name="dev-final")
    test_metrics, test_preds = run_eval(model, test_loader, test_data, args, device, split_name="test-final")

    save_json(output_dir / "dev_predictions.json", dev_preds, pretty=True)
    save_json(output_dir / "test_predictions.json", test_preds, pretty=True)
    save_json(output_dir / "dev_metrics.json", dev_metrics, pretty=True)
    save_json(output_dir / "test_metrics.json", test_metrics, pretty=True)
    dev_table_paths = save_metric_tables(output_dir, "dev", dev_metrics)
    test_table_paths = save_metric_tables(output_dir, "test", test_metrics)
    print(f"[Output] predictions, metrics, and table files saved to {output_dir}")
    print(f"[Output] dev tables: {json.dumps(dev_table_paths, ensure_ascii=False)}")
    print(f"[Output] test tables: {json.dumps(test_table_paths, ensure_ascii=False)}")

    print("BEST_DEV_F1", round(best_dev, 4))
    print(format_metric_output("dev", dev_metrics))
    print(format_metric_output("test", test_metrics))
    print(f"[Done] total_elapsed={format_elapsed(time.time() - job_start)}")


def join_dataset_file(dataset_dir, file_name):
    if not file_name:
        raise ValueError("Config missing one of datamodule.train_file/dev_file/test_file")
    path = Path(str(file_name))
    if path.is_absolute():
        return str(path)
    return str(Path(str(dataset_dir)) / path)


def namespace_from_hydra_config(cfg, config_dir):
    cfg_dict = cfg if isinstance(cfg, dict) else cfg
    datamodule = cfg_dict.get("datamodule", {})
    model_cfg = cfg_dict.get("model", {})
    train_cfg = cfg_dict.get("train", {})
    output_cfg = cfg_dict.get("output", {})

    dataset_dir = datamodule.get("dataset_dir")
    if not dataset_dir:
        raise ValueError("Config missing datamodule.dataset_dir")

    device = str(train_cfg.get("device", "cuda:0"))
    return argparse.Namespace(
        config="hydra",
        train_path=join_dataset_file(dataset_dir, datamodule.get("train_file")),
        dev_path=join_dataset_file(dataset_dir, datamodule.get("dev_file")),
        test_path=join_dataset_file(dataset_dir, datamodule.get("test_file")),
        output_dir=resolve_path(output_cfg.get("output_dir", "outputs/doc_run"), config_dir),
        model_name=resolve_model_name(model_cfg.get("model_name_or_path", "bert-base-uncased"), config_dir),
        max_length=int(model_cfg.get("max_seq_length", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        threshold=float(model_cfg.get("threshold", 0.5)),
        loss_type=str(model_cfg.get("loss_type", "bce")),
        neg_ratio=float(datamodule.get("neg_ratio", model_cfg.get("neg_ratio", 3.0))),
        epochs=int(train_cfg.get("epochs", 3)),
        log_steps=int(train_cfg.get("log_steps", 100)),
        eval_test_each_epoch=bool(train_cfg.get("eval_test_each_epoch", True)),
        batch_size=int(datamodule.get("train_batch_size", 2)),
        eval_batch_size=int(datamodule.get("test_batch_size", 2)),
        lr=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        seed=int(train_cfg.get("seed", 13)),
        device=device,
        cpu=device.lower() == "cpu",
    )





