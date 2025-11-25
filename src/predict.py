
# Updated predict.py — keeps learned model as main detector, adds confidence threshold
import json
import argparse
import os
import re
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d -]{6,}\d)\b")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

def bio_to_spans_with_conf(text: str, offsets: List[Tuple[int,int]], label_ids: List[int], confidences: List[float]):
    spans = []
    current_label = None
    current_start = None
    current_end = None
    confs = []

    for (start, end), lid, conf in zip(offsets, label_ids, confidences):
        # offsets may include (0,0) for special tokens
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                spans.append((current_start, current_end, current_label, avg_conf))
                current_label = None
                confs = []
            continue

        if "-" not in label:
            # safety
            if current_label is not None:
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                spans.append((current_start, current_end, current_label, avg_conf))
                current_label = None
                confs = []
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                spans.append((current_start, current_end, current_label, avg_conf))
            current_label = ent_type
            current_start = start
            current_end = end
            confs = [conf]
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                confs.append(conf)
            else:
                if current_label is not None:
                    avg_conf = sum(confs) / len(confs) if confs else 0.0
                    spans.append((current_start, current_end, current_label, avg_conf))
                current_label = ent_type
                current_start = start
                current_end = end
                confs = [conf]

    if current_label is not None:
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        spans.append((current_start, current_end, current_label, avg_conf))

    return spans

def regex_verify(span_text: str, label: str) -> bool:
    t = span_text.strip()
    if label == "EMAIL":
        return bool(EMAIL_RE.search(t))
    if label == "PHONE":
        return bool(PHONE_RE.search(re.sub(r"[()\\s-]+", "", t)))
    if label == "CREDIT_CARD":
        return bool(CC_RE.search(re.sub(r"[ -]+", "", t)))
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--conf_thresh", type=float, default=0.55, help="token-level avg confidence threshold for accepting entity spans")
    ap.add_argument("--min_span_len", type=int, default=2)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]  # seq_len x num_labels
                probs = F.softmax(logits, dim=-1)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

                confidences = []
                for i, pid in enumerate(pred_ids):
                    confidences.append(float(probs[i, pid].cpu().item()))

            spans = bio_to_spans_with_conf(text, offsets, pred_ids, confidences)
            ents = []
            for s, e, lab, conf in spans:
                if e - s < args.min_span_len:
                    continue
                if conf < args.conf_thresh:
                    continue
                span_text = text[s:e]
                if label_is_pii(lab):
                    if lab in {"EMAIL", "PHONE", "CREDIT_CARD"}:
                        if not regex_verify(span_text, lab):
                            continue
                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")

if __name__ == "__main__":
    main()
