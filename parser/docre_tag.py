# -*- coding: utf-8 -*-
"""Discriminative DocRE-style baseline: sequence labelling on the same backbone.
Each token of the epitaph is tagged with the canonical path code of the relative it belongs to (B-/I- per code, O otherwise),
with Qwen2.5-7B + LoRA and a token-classification head (Qwen2ForTokenClassification). Decoding merges B/I spans into
(name, path) pairs and writes runs/<name>/pred_<split>.jsonl in the parser's format, so the same metrics apply.
Usage: python docre_tag.py --name TAG --train sft/G_train.jsonl --evals sft/G_dev.jsonl sft/G_test.jsonl [--epochs 3] [--seed 0]
"""
import os, sys, json, math, random, argparse, time, collections, re
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoTokenizer, Qwen2ForTokenClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--train", required=True); ap.add_argument("--evals", nargs="+", default=[])
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--maxlen", type=int, default=2048); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-only", action="store_true")
a = ap.parse_args(); OUT = os.path.join(HERE, "runs", a.name); os.makedirs(OUT, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
log = open(os.path.join(OUT, "train.log"), "a", encoding="utf-8")
def P(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); log.write(s + "\n"); log.flush()
def doc_text(d): return d["prompt"].split("墓志:\n", 1)[1] if "墓志:\n" in d["prompt"] else d["prompt"]
def rows(t): return [tuple(x.split(" | ")) for x in t.splitlines() if " | " in x]

# ---------- label set: codes with >= 5 training occurrences; rarer codes map to O (cannot be predicted)
train = [json.loads(l) for l in open(os.path.join(HERE, a.train), encoding="utf-8")]
cnt = collections.Counter(p for d in train for _, p in rows(d["target"]))
CODES = sorted([c for c, n in cnt.items() if n >= 5], key=lambda c: -cnt[c]); LAB = ["O"] + ["B-" + c for c in CODES] + ["I-" + c for c in CODES]; L2I = {l: i for i, l in enumerate(LAB)}
P("args", vars(a)); P("codes", len(CODES), "labels", len(LAB), "coverage %.1f%%" % (100 * sum(cnt[c] for c in CODES) / sum(cnt.values())))

tok = AutoTokenizer.from_pretrained(a.base)
def char_labels(d):
    """character-level tags from (name, path): tag the first occurrence of the full name, else of the given name (surname omitted in text)"""
    text = doc_text(d); tags = ["O"] * len(text)
    for n, p in rows(d["target"]):
        if p not in cnt or cnt[p] < 5 or not n: continue
        # candidates: full name; given name (surname omitted in the text); for two-character names the single given
        # character only where it follows a naming marker (讳/名/曰/字), which is how epitaphs introduce relatives
        cands = [(n, None)] + ([(n[1:], None)] if len(n) >= 3 else [(n[1], "讳名曰字祖父子弟兄姊妹女孙侄甥婿妻考妣嗣")])
        for cand, after in cands:
            if not cand: continue
            i = text.find(cand)
            while i >= 0 and (any(t != "O" for t in tags[i:i + len(cand)]) or (after and (i == 0 or text[i - 1] not in after))): i = text.find(cand, i + 1)
            if i >= 0:
                tags[i] = "B-" + p
                for k in range(i + 1, i + len(cand)): tags[k] = "I-" + p
                break
    return text, tags
def encode(text, tags=None):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=a.maxlen)
    ids = enc["input_ids"]; off = enc["offset_mapping"]; y = None
    if tags is not None:
        y = []
        for (s, e) in off:
            if e <= s: y.append(-100); continue
            t = tags[s]
            if t.startswith("I-") and s > 0 and tags[s - 1] == "O": t = "B-" + t[2:]     # token starts mid-span after an untagged char: treat as B
            y.append(L2I.get(t, 0))
    return ids, off, y

model = Qwen2ForTokenClassification.from_pretrained(a.base, num_labels=len(LAB), torch_dtype=torch.bfloat16, device_map=("auto" if os.environ.get("DEVMAP") == "auto" else {"": 0}), max_memory=(eval(os.environ["MAXMEM"]) if os.environ.get("MAXMEM") else None))
model.config.pad_token_id = tok.pad_token_id
if a.eval_only:
    model = PeftModel.from_pretrained(model, OUT)
else:
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="TOKEN_CLS", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], modules_to_save=["score"]))
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    data = [char_labels(d) for d in train]; data = [(t, g) for t, g in data if t]
    n_tagged = sum(1 for _, g in data for x in g if x.startswith("B-")); P("train docs", len(data), "tagged spans", n_tagged, "of", sum(len(rows(d["target"])) for d in train))
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(data) * a.epochs / a.accum); sch = get_cosine_schedule_with_warmup(opt, int(total * 0.05), total)
    model.train(); step = 0; t0 = time.time(); run = 0.0; n_seen = 0; n_target = int(len(data) * a.epochs)
    dev0 = next(model.parameters()).device
    while n_seen < n_target:
        random.shuffle(data)
        for text, tags in data:
            if n_seen >= n_target: break
            ids, _, y = encode(text, tags)
            logits = model(input_ids=torch.tensor([ids]).to(dev0)).logits[0]
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(y).to(logits.device), ignore_index=-100)
            (loss / a.accum).backward(); run += loss.item(); n_seen += 1
            if n_seen % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
                if step % 20 == 0: P("step %d/%d loss %.4f  %.0fs" % (step, total, run / (20 * a.accum), time.time() - t0)); run = 0.0
    model.save_pretrained(OUT); P("TRAIN_DONE", "%.0fs" % (time.time() - t0))

# ---------- decode + evaluate
model.eval(); dev0 = next(model.parameters()).device
def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))
for ev in a.evals:
    split = os.path.basename(ev).split("_")[-1].split(".")[0]; docs = [json.loads(l) for l in open(os.path.join(HERE, ev), encoding="utf-8")]
    out = []; hit = tot = 0; n_out = 0
    with torch.no_grad():
        for d in docs:
            text = doc_text(d); ids, off, _ = encode(text)
            logits = model(input_ids=torch.tensor([ids]).to(dev0)).logits[0]; pred = logits.argmax(-1).tolist()
            spans = []; cur = None
            for (s, e), li in zip(off, pred):
                lab = LAB[li]
                if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur[2] != lab[2:])):
                    if cur: spans.append(cur)
                    cur = [s, e, lab[2:]]
                elif lab.startswith("I-") and cur: cur[1] = e
                else:
                    if cur: spans.append(cur); cur = None
            if cur: spans.append(cur)
            preds = {}
            for s, e, code in spans:
                name = text[s:e].strip("，。、；：「」『』（）()《》 \n")
                if 1 <= len(name) <= 6 and name not in preds: preds[name] = code
            # surname completion as the parser's evaluation does: single-char given names get the ego's surname when the text omits it
            root = d["root"]; sur = root[0] if len(root) >= 2 else ""
            preds2 = {}
            for n, c in preds.items():
                full = (sur + n) if (sur and len(n) == 1) else n            # single given character -> ego's surname; X氏 and longer names kept as written
                preds2.setdefault(full, c)
            out.append({"id": d["id"], "pred": preds2}); n_out += len(preds2)
            gold = rows(d["target"]); tot += len(gold)
            for n, p in gold:
                if any((n == m or (len(n) >= 3 and n[1:] == m) or m.endswith(n[1:] if len(n) >= 3 else n)) and same(c, p) for m, c in preds2.items()): hit += 1
    with open(os.path.join(OUT, "pred_%s.jsonl" % split), "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    msg = "%s  fmt=TAG  %s  文档 %d\n召回 %.1f%% (%d/%d); 输出 %d 人" % (a.name, split, len(docs), 100 * hit / max(1, tot), hit, tot, n_out)
    open(os.path.join(OUT, "eval_%s.txt" % split), "w", encoding="utf-8").write(msg); P(msg)
P("RUN_DONE")
