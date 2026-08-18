import argparse
import json
import random
from pathlib import Path

from common import REL_ID_TO_NAME, REL_NAME_TO_ID, evaluate_triples, load_json, save_json


def entity_name(doc, idx):
    name = doc["vertexSet"][idx][0].get("name", "")
    if isinstance(name, list):
        return " ".join(str(x) for x in name)
    return str(name)


def entity_type(doc, idx):
    return str(doc["vertexSet"][idx][0].get("type", ""))


def doc_text(doc, max_sentences=None):
    sents = doc.get("sents", [])
    if max_sentences:
        sents = sents[:max_sentences]
    lines = []
    for i, sent in enumerate(sents):
        lines.append(f"[S{i}] " + " ".join(sent))
    return "\n".join(lines)


def schema_text():
    return "\n".join(f"{rid}: {name}" for rid, name in REL_ID_TO_NAME.items())


def entity_list(doc):
    lines = []
    for i, mentions in enumerate(doc.get("vertexSet", [])):
        lines.append(f"{i}: {entity_name(doc, i)} ({entity_type(doc, i)})")
    return "\n".join(lines)


def gold_for_doc(doc):
    rows = []
    for label in doc.get("labels", []):
        r = int(label["r"]) if str(label.get("r")).isdigit() else REL_NAME_TO_ID[label["r"]]
        rows.append(
            {
                "h": int(label["h"]),
                "t": int(label["t"]),
                "r": r,
                "relation": REL_ID_TO_NAME[r],
            }
        )
    return rows


def make_prompt(doc):
    return f"""You are performing maritime document-level relation extraction.

Return only a JSON array. Each item must have integer fields "h", "t", and "r".
Use only the entity IDs and relation IDs listed below. Do not invent entities or relations.
Predict a relation only if it is supported by the document.

Relation schema:
{schema_text()}

Entities:
{entity_list(doc)}

Document:
{doc_text(doc)}

JSON output:
"""


def make_prompts(args):
    data = load_json(args.data_path)
    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[: args.num_docs]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in indices:
            doc = data[i]
            row = {
                "doc_index": i,
                "title": doc.get("title"),
                "prompt": make_prompt(doc),
                "gold": gold_for_doc(doc),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote prompts: {out_path}")


def parse_prediction_value(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```"):
            value = value.strip("`")
            value = value.replace("json", "", 1).strip()
        return json.loads(value)
    raise ValueError(f"Unsupported prediction value: {type(value)}")


def evaluate_outputs(args):
    data = load_json(args.data_path)
    pred_triples = []
    details = []
    with open(args.prediction_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_i = int(row["doc_index"])
            title = data[doc_i].get("title")
            raw = row.get("prediction", row.get("output", row.get("response", [])))
            try:
                items = parse_prediction_value(raw)
            except Exception as exc:
                details.append({"doc_index": doc_i, "title": title, "parse_error": str(exc), "raw": raw})
                continue
            for item in items:
                try:
                    h, t = int(item["h"]), int(item["t"])
                    r_value = item["r"]
                    r = int(r_value) if str(r_value).isdigit() else REL_NAME_TO_ID[str(r_value)]
                    pred_triples.append((title, h, t, r))
                except Exception as exc:
                    details.append({"doc_index": doc_i, "title": title, "item_error": str(exc), "item": item})

    sampled_doc_indices = []
    with open(args.prediction_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sampled_doc_indices.append(int(json.loads(line)["doc_index"]))
    sampled_data = [data[i] for i in sampled_doc_indices]

    metrics = evaluate_triples(sampled_data, pred_triples)
    save_json(args.metrics_path, metrics, pretty=True)
    if args.error_path:
        save_json(args.error_path, details, pretty=True)
    print(json.dumps({k: metrics[k] for k in ["p", "r", "f1", "gold", "pred", "correct"]}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-prompts")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--num_docs", type=int, default=100)
    p.add_argument("--seed", type=int, default=13)

    e = sub.add_parser("eval")
    e.add_argument("--data_path", required=True)
    e.add_argument("--prediction_path", required=True)
    e.add_argument("--metrics_path", required=True)
    e.add_argument("--error_path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "make-prompts":
        make_prompts(args)
    elif args.cmd == "eval":
        evaluate_outputs(args)
