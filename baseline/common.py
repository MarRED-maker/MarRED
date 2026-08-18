import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


REL_ID_TO_NAME = {
    1: "has_facility",
    2: "handles_cargo",
    3: "flagged_in",
    4: "part_of",
    5: "has_deficiency",
    6: "classified_by",
    7: "located_in",
    8: "owned_by",
    9: "operated_by",
    10: "calls_at",
}
REL_NAME_TO_ID = {v: k for k, v in REL_ID_TO_NAME.items()}
REL_IDS = sorted(REL_ID_TO_NAME)
REL_ID_TO_INDEX = {r: i for i, r in enumerate(REL_IDS)}
REL_INDEX_TO_ID = {i: r for r, i in REL_ID_TO_INDEX.items()}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj, pretty=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def rel_id(label):
    value = label.get("r")
    if isinstance(value, int):
        return value
    if str(value).isdigit():
        return int(value)
    return REL_NAME_TO_ID[value]


def mention_text(mention):
    name = mention.get("name", "")
    if isinstance(name, list):
        return " ".join(str(x) for x in name)
    return str(name)


def entity_name(doc, ent_idx):
    return mention_text(doc["vertexSet"][ent_idx][0])


def entity_type(doc, ent_idx):
    return str(doc["vertexSet"][ent_idx][0].get("type", ""))


def sentence_text(sent):
    return " ".join(str(x) for x in sent)


def gold_relation_map(doc):
    pair_to_rels = defaultdict(set)
    for label in doc.get("labels", []):
        pair_to_rels[(int(label["h"]), int(label["t"]))].add(rel_id(label))
    return pair_to_rels


def dist_to_scope(value):
    text = str(value).upper()
    if text == "CROSS" or text == "1":
        return "inter"
    if text == "NON-CROSS" or text == "0":
        return "intra"
    return None


def pair_scope(doc, h, t):
    """Return intra/inter for an entity pair, following the TTM dist convention.

    Gold labels store dist as CROSS/NON-CROSS. For false-positive predicted
    triples not present in gold labels, fall back to mention sentence overlap so
    precision can still be split by intra/inter.
    """
    for label in doc.get("labels", []):
        if int(label.get("h", -1)) == int(h) and int(label.get("t", -1)) == int(t):
            scope = dist_to_scope(label.get("dist"))
            if scope is not None:
                return scope

    h_sents = {int(m["sent_id"]) for m in doc.get("vertexSet", [])[h]}
    t_sents = {int(m["sent_id"]) for m in doc.get("vertexSet", [])[t]}
    return "intra" if h_sents & t_sents else "inter"


def label_vector(rel_ids):
    vec = [0.0] * len(REL_IDS)
    for r in rel_ids:
        if r in REL_ID_TO_INDEX:
            vec[REL_ID_TO_INDEX[r]] = 1.0
    return vec


def all_ordered_entity_pairs(doc):
    n = len(doc.get("vertexSet", []))
    for h in range(n):
        for t in range(n):
            if h != t:
                yield h, t


def first_same_sentence_mentions(doc, h, t):
    """Return (sent_id, h_mention, t_mention) if the pair co-occurs in a sentence."""
    h_mentions = doc["vertexSet"][h]
    t_mentions = doc["vertexSet"][t]
    for hm in h_mentions:
        hs = int(hm["sent_id"])
        for tm in t_mentions:
            if hs == int(tm["sent_id"]):
                return hs, hm, tm
    return None


def closest_mentions(doc, h, t):
    """Pick the closest mention pair in document token space."""
    best = None
    best_dist = 10**9
    for hm in doc["vertexSet"][h]:
        for tm in doc["vertexSet"][t]:
            hp = hm.get("global_pos")
            tp = tm.get("global_pos")
            if hp is None or tp is None:
                hs, ts = int(hm["sent_id"]), int(tm["sent_id"])
                dist = abs(hs - ts) * 1000 + abs(int(hm["pos"][0]) - int(tm["pos"][0]))
            else:
                dist = abs(int(hp[0]) - int(tp[0]))
            if dist < best_dist:
                best = (hm, tm)
                best_dist = dist
    return best


def insert_markers(tokens, spans):
    """Insert entity markers around token spans.

    spans is a list of (start, end, start_marker, end_marker). End is exclusive.
    """
    events = defaultdict(list)
    for start, end, start_marker, end_marker in spans:
        events[int(start)].append(start_marker)
        events[int(end)].append(end_marker)
    out = []
    for i, tok in enumerate(tokens):
        out.extend(events.get(i, []))
        out.append(tok)
    out.extend(events.get(len(tokens), []))
    return out


def sentence_pair_context(doc, h, t):
    found = first_same_sentence_mentions(doc, h, t)
    if found is None:
        return None
    sent_id, hm, tm = found
    tokens = list(doc["sents"][sent_id])
    spans = [
        (hm["pos"][0], hm["pos"][1], "[E1]", "[/E1]"),
        (tm["pos"][0], tm["pos"][1], "[E2]", "[/E2]"),
    ]
    return " ".join(insert_markers(tokens, spans))


def document_pair_context(doc, h, t, max_words=700):
    hm, tm = closest_mentions(doc, h, t)
    flat_tokens = []
    offsets = []
    cursor = 0
    for sent in doc.get("sents", []):
        offsets.append(cursor)
        flat_tokens.extend(sent)
        cursor += len(sent)
        flat_tokens.append("[SENT]")
        cursor += 1

    def global_span(m):
        sent_offset = offsets[int(m["sent_id"])]
        return sent_offset + int(m["pos"][0]), sent_offset + int(m["pos"][1])

    h_start, h_end = global_span(hm)
    t_start, t_end = global_span(tm)

    if len(flat_tokens) > max_words:
        left = min(h_start, t_start)
        right = max(h_end, t_end)
        center = (left + right) // 2
        window_start = max(0, center - max_words // 2)
        window_end = min(len(flat_tokens), window_start + max_words)
        window_start = max(0, window_end - max_words)
        flat_tokens = flat_tokens[window_start:window_end]
        h_start, h_end = h_start - window_start, h_end - window_start
        t_start, t_end = t_start - window_start, t_end - window_start
        if h_start < 0 or t_start < 0 or h_end > len(flat_tokens) or t_end > len(flat_tokens):
            return None

    spans = [
        (h_start, h_end, "[E1]", "[/E1]"),
        (t_start, t_end, "[E2]", "[/E2]"),
    ]
    return " ".join(insert_markers(flat_tokens, spans))


def build_pair_records(data, mode, include_negative=True, neg_ratio=3.0, seed=13, max_words=700, desc=None):
    rng = random.Random(seed)
    records = []
    iterator = tqdm(data, desc=desc or f"build {mode} train records", total=len(data), dynamic_ncols=True)
    for doc_i, doc in enumerate(iterator):
        gold = gold_relation_map(doc)
        positives = set(gold)
        negatives = []
        for h, t in all_ordered_entity_pairs(doc):
            rels = gold.get((h, t), set())
            context = (
                sentence_pair_context(doc, h, t)
                if mode == "sentence"
                else document_pair_context(doc, h, t, max_words=max_words)
            )
            if context is None:
                continue
            rec = {
                "doc_i": doc_i,
                "title": doc.get("title", str(doc_i)),
                "h": h,
                "t": t,
                "head": entity_name(doc, h),
                "tail": entity_name(doc, t),
                "context": context,
                "labels": label_vector(rels),
                "is_positive": bool(rels),
            }
            if rels:
                records.append(rec)
            else:
                negatives.append(rec)
        if include_negative and negatives:
            if neg_ratio < 0:
                records.extend(negatives)
            else:
                keep = min(len(negatives), int(max(1, len(positives)) * neg_ratio))
                records.extend(rng.sample(negatives, keep))
    rng.shuffle(records)
    return records

def build_eval_records(data, mode, max_words=700, desc=None):
    records = []
    iterator = tqdm(data, desc=desc or f"build {mode} eval records", total=len(data), dynamic_ncols=True)
    for doc_i, doc in enumerate(iterator):
        gold = gold_relation_map(doc)
        for h, t in all_ordered_entity_pairs(doc):
            context = (
                sentence_pair_context(doc, h, t)
                if mode == "sentence"
                else document_pair_context(doc, h, t, max_words=max_words)
            )
            if context is None:
                continue
            records.append(
                {
                    "doc_i": doc_i,
                    "title": doc.get("title", str(doc_i)),
                    "h": h,
                    "t": t,
                    "head": entity_name(doc, h),
                    "tail": entity_name(doc, t),
                    "context": context,
                    "labels": label_vector(gold.get((h, t), set())),
                }
            )
    return records

def gold_triples(data):
    triples = set()
    by_relation = Counter()
    for doc in data:
        title = doc.get("title", "")
        for label in doc.get("labels", []):
            triple = (title, int(label["h"]), int(label["t"]), rel_id(label))
            triples.add(triple)
            by_relation[rel_id(label)] += 1
    return triples, by_relation


def fact_scope_map(data):
    scopes = {}
    doc_by_title = {}
    for doc in data:
        title = doc.get("title", "")
        doc_by_title[title] = doc
        for label in doc.get("labels", []):
            h, t, r = int(label["h"]), int(label["t"]), rel_id(label)
            fact = (title, h, t, r)
            scopes[fact] = dist_to_scope(label.get("dist")) or pair_scope(doc, h, t)
    return scopes, doc_by_title


def metric_from_sets(gold, pred):
    correct = gold & pred
    p = len(correct) / len(pred) * 100 if pred else 0.0
    r = len(correct) / len(gold) * 100 if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "p": p,
        "r": r,
        "f1": f1,
        "gold": len(gold),
        "pred": len(pred),
        "correct": len(correct),
    }


def evaluate_triples(data, pred_triples):
    gold, _ = gold_triples(data)
    pred = set(pred_triples)
    overall = metric_from_sets(gold, pred)
    scopes, doc_by_title = fact_scope_map(data)

    per_rel = {}
    for rel in REL_IDS:
        g = {x for x in gold if x[3] == rel}
        pr = {x for x in pred if x[3] == rel}
        per_rel[REL_ID_TO_NAME[rel]] = metric_from_sets(g, pr)

    pred_scope = {}
    for fact in pred:
        if fact in scopes:
            pred_scope[fact] = scopes[fact]
            continue
        title, h, t, _ = fact
        doc = doc_by_title.get(title)
        pred_scope[fact] = pair_scope(doc, h, t) if doc is not None else "unknown"

    by_scope = {}
    for scope in ["intra", "inter"]:
        g = {fact for fact in gold if scopes.get(fact) == scope}
        pr = {fact for fact in pred if pred_scope.get(fact) == scope}
        by_scope[scope] = metric_from_sets(g, pr)

    return {
        "p": overall["p"],
        "r": overall["r"],
        "f1": overall["f1"],
        "gold": overall["gold"],
        "pred": overall["pred"],
        "correct": overall["correct"],
        "per_relation": per_rel,
        "intra": by_scope["intra"],
        "inter": by_scope["inter"],
        "f1_intra": by_scope["intra"]["f1"],
        "f1_inter": by_scope["inter"]["f1"],
    }


def print_scope_metrics(tag, metrics):
    intra = metrics.get("intra", {})
    inter = metrics.get("inter", {})
    print(
        f"[Eval] split={tag} intra_p={intra.get('p', 0.0):.2f} "
        f"intra_r={intra.get('r', 0.0):.2f} intra_f1={intra.get('f1', 0.0):.2f} "
        f"inter_p={inter.get('p', 0.0):.2f} inter_r={inter.get('r', 0.0):.2f} "
        f"inter_f1={inter.get('f1', 0.0):.2f}"
    )


def print_per_relation_metrics(tag, metrics):
    print(f"[Eval] split={tag} per-relation metrics")
    print("relation\tP\tR\tF1\tgold\tpred\tcorrect")
    for rel in REL_ID_TO_NAME.values():
        row = metrics.get("per_relation", {}).get(rel, {})
        print(
            f"{rel}\t{row.get('p', 0.0):.2f}\t{row.get('r', 0.0):.2f}\t"
            f"{row.get('f1', 0.0):.2f}\t{row.get('gold', 0)}\t"
            f"{row.get('pred', 0)}\t{row.get('correct', 0)}"
        )


def save_metric_tables(output_dir, tag, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_path = output_dir / f"{tag}_per_relation_metrics.tsv"
    with open(per_path, "w", encoding="utf-8") as f:
        f.write("relation\tP\tR\tF1\tgold\tpred\tcorrect\n")
        for rel in REL_ID_TO_NAME.values():
            row = metrics.get("per_relation", {}).get(rel, {})
            f.write(
                f"{rel}\t{row.get('p', 0.0):.4f}\t{row.get('r', 0.0):.4f}\t"
                f"{row.get('f1', 0.0):.4f}\t{row.get('gold', 0)}\t"
                f"{row.get('pred', 0)}\t{row.get('correct', 0)}\n"
            )

    scope_path = output_dir / f"{tag}_intra_inter_metrics.tsv"
    with open(scope_path, "w", encoding="utf-8") as f:
        f.write("scope\tP\tR\tF1\tgold\tpred\tcorrect\n")
        for scope in ["intra", "inter"]:
            row = metrics.get(scope, {})
            f.write(
                f"{scope}\t{row.get('p', 0.0):.4f}\t{row.get('r', 0.0):.4f}\t"
                f"{row.get('f1', 0.0):.4f}\t{row.get('gold', 0)}\t"
                f"{row.get('pred', 0)}\t{row.get('correct', 0)}\n"
            )

    latex_rel_path = output_dir / f"{tag}_per_relation_table.tex"
    with open(latex_rel_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Relation & P & R & F1 \\\n")
        f.write("\\midrule\n")
        for rel in REL_ID_TO_NAME.values():
            row = metrics.get("per_relation", {}).get(rel, {})
            rel_latex = rel.replace("_", r"\_")
            f.write(
                f"{rel_latex} & {row.get('p', 0.0):.2f} & "
                f"{row.get('r', 0.0):.2f} & {row.get('f1', 0.0):.2f} \\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    latex_scope_path = output_dir / f"{tag}_intra_inter_table_rows.tex"
    with open(latex_scope_path, "w", encoding="utf-8") as f:
        intra = metrics.get("intra", {})
        inter = metrics.get("inter", {})
        f.write(f"{tag} & {intra.get('f1', 0.0):.2f} & {inter.get('f1', 0.0):.2f} \\\n")

    return {
        "per_relation_tsv": str(per_path),
        "intra_inter_tsv": str(scope_path),
        "per_relation_tex": str(latex_rel_path),
        "intra_inter_tex_rows": str(latex_scope_path),
    }


def predictions_from_scores(records, score_rows, threshold=0.5):
    triples = []
    details = []
    for rec, scores in zip(records, score_rows):
        rels = []
        for i, score in enumerate(scores):
            if float(score) >= threshold:
                rel_id_value = REL_INDEX_TO_ID[i]
                triples.append((rec["title"], int(rec["h"]), int(rec["t"]), rel_id_value))
                rels.append({"r": rel_id_value, "relation": REL_ID_TO_NAME[rel_id_value], "score": float(score)})
        if rels:
            details.append(
                {
                    "title": rec["title"],
                    "h": int(rec["h"]),
                    "t": int(rec["t"]),
                    "head": rec.get("head"),
                    "tail": rec.get("tail"),
                    "relations": rels,
                }
            )
    return triples, details



