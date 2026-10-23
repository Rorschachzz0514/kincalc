# -*- coding: utf-8 -*-
"""
用微调后的模型在 dev/test 上生成并评测。
用法: python infer_eval.py --name G --fmt G --data sft/G_dev.jsonl [--vote] [--base-only]
  --fmt G  输出 "姓名 | 路径"            --fmt P  输出 "姓名 | 参照人 | 局部路径", 由代数沿参照链组合
  --vote   换根投票 (只对 G 格式且模型见过换根样本有意义): 另以父/配偶/第一个子女为中心各解析一次,
           把结果换根回墓主视角, 与直接解析的结果多数表决
指标: 按金标路径跳数的召回 (名字找到且路径正确), 以及预测能对上金标者的路径正确率。
输出: runs/<name>/pred_<split>[_vote].jsonl, eval_<split>[_vote].txt
"""
import os, sys, json, argparse, collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
from kin_algebra import normalize, reroot, inverse, BASE, GENDER_OF, view_consistent, view_to_ego

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--fmt", default="G"); ap.add_argument("--data", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--vote", action="store_true"); ap.add_argument("--base-only", action="store_true"); ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
OUT = os.path.join(HERE, "runs", a.name); os.makedirs(OUT, exist_ok=True)
split = os.path.basename(a.data).replace(".jsonl", "").replace(a.fmt + "_", "", 1) + ("_vote" if a.vote else "")
data = [json.loads(l) for l in open(os.path.join(HERE, a.data), encoding="utf-8")]

tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map=("auto" if os.environ.get("DEVMAP") == "auto" else {"": 0}), max_memory=(eval(os.environ["MAXMEM"]) if os.environ.get("MAXMEM") else None))
if not a.base_only: model = PeftModel.from_pretrained(model, OUT)
model.eval()


@torch.no_grad()
def generate(prompts):
    outs = []
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i])); res = {}
    for b in range(0, len(order), a.bs):
        idx = order[b:b + a.bs]
        txt = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True) for i in idx]
        x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to(next(model.parameters()).device)
        y = model.generate(**x, max_new_tokens=512, do_sample=False, repetition_penalty=1.0)
        for i, o in zip(idx, y): res[i] = tok.decode(o[x["input_ids"].shape[1]:], skip_special_tokens=True)
    return [res[i] for i in range(len(prompts))]


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


def match(pred, gd):
    """金标带 aliases (英文数据) 时按别名精确匹配 (不分大小写), 否则按中文姓名规则"""
    if "aliases" in gd: return pred.lower() in {x.lower() for x in gd["aliases"]}
    return same(pred, gd["name"])


def parse_G(out, g):
    res = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) == 2 and p[1] and set(p[1]) <= BASE and p[0] not in res: res[p[0]] = normalize(p[1], g)
    return res


def parse_P(out, g):
    rows = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) == 3 and p[2] and set(p[2]) <= BASE and p[0] not in rows: rows[p[0]] = (p[1], p[2])
    res = {}
    def resolve(name, depth=0):
        if name in res: return res[name]
        if name not in rows or depth > 8: return None
        an, r = rows[name]
        base = "" if an == "中心" else (resolve(an, depth + 1) if an != name else None)
        if base is None: return None
        res[name] = normalize(base + r, g); return res[name]
    for k in rows: resolve(k)
    return res


outs = generate([d["prompt"] for d in data])
preds = [(parse_G if a.fmt == "G" else parse_P)(o, d["g_root"]) for o, d in zip(outs, data)]

vstat = collections.Counter()
if a.vote:
    # 选新中心: 直接解析结果里的 父(F) / 配偶(W,H) / 第一个子(S) / 祖父(FF)
    jobs = []
    for di, (d, pr) in enumerate(zip(data, preds)):
        used = set()
        for name, path in pr.items():
            if path in ("F", "W", "H", "S", "FF") and path not in used and len(name) >= 2:
                used.add(path); jobs.append((di, name, path))
    P0 = data[0]["prompt"]; head = P0[:P0.index("「")]; tail_tpl = lambda d: d["prompt"][d["prompt"].index("」"):]
    v_outs = generate([head + "「" + name + tail_tpl(data[di]) for di, name, _ in jobs])
    votes = [collections.defaultdict(collections.Counter) for _ in data]
    for di, pr in enumerate(preds):
        for n, p in pr.items(): votes[di][n][p] += 1.5                       # 直接解析权重略高, 平票时取它
    for (di, yname, ypath), o in zip(jobs, v_outs):
        g_ego = data[di]["g_root"]; g_y = GENDER_OF[ypath[-1]]
        view = parse_G(o, g_y)
        for n, p in view.items():
            if same(n, data[di]["root"]) or same(data[di]["root"], n): continue
            key = next((k for k in votes[di] if same(n, k) or same(k, n)), None)
            if key is not None and key in preds[di] and view_consistent(ypath, preds[di][key], p, g_ego, g_y):
                votes[di][key][preds[di][key]] += 1; vstat["view_votes"] += 1; continue      # 该视角印证直接结果
            cand = view_to_ego(ypath, p, g_ego, g_y)[:1]                                       # 否则给出它自己的读法
            if cand: votes[di][key if key is not None else n][cand[0]] += 1; vstat["view_votes"] += 1
    new = []
    for di, v in enumerate(votes):
        r = {}
        for n, c in v.items():
            best, cnt = c.most_common(1)[0]
            if n not in preds[di] and cnt < 2: continue                                        # 直接视角没有的人: 至少两个视角一致才补
            r[n] = best
            if n in preds[di] and preds[di][n] != best: vstat["changed"] += 1
            if n not in preds[di]: vstat["added"] += 1
        new.append(r)
    preds = new; vstat["views"] = len(jobs)

NEU = lambda p: p.translate(str.maketrans("FMSDHWBZ", "PPCCEEGG"))      # 不分性别比较 (英文数据的金标多为不分性别的码)
by = collections.defaultdict(collections.Counter); tot = collections.Counter(); errs = collections.Counter()
with open(os.path.join(OUT, "pred_%s.jsonl" % split), "w", encoding="utf-8") as f:
    for d, o, pr in zip(data, outs, preds):
        f.write(json.dumps(dict(id=d["id"], out=o, pred=pr, gold=d["gold"]), ensure_ascii=False) + "\n")
        tot["pred"] += len(pr)
        for gd in d["gold"]:
            L = min(len(gd["path"]), 4); by[L]["n"] += 1
            hit = [p for k, p in pr.items() if match(k, gd)]
            by[L]["found"] += bool(hit); ok = gd["path"] in hit; by[L]["ok"] += ok
            tot["ok_neutral"] += NEU(gd["path"]) in [NEU(h) for h in hit]
            if hit and not ok: errs[(gd["path"], hit[0])] += 1
        for k, p in pr.items():
            gm = [gd["path"] for gd in d["gold"] if match(k, gd)]
            if gm: tot["matched"] += 1; tot["matched_ok"] += p in gm
n = sum(c["n"] for c in by.values()); ok = sum(c["ok"] for c in by.values()); fd = sum(c["found"] for c in by.values())
S = ["%s  fmt=%s  %s  文档 %d%s" % (a.name, a.fmt, split, len(data), "  (未微调)" if a.base_only else ""),
     "召回 %.1f%% (%d/%d); 名字找到率 %.1f%%; 找到者中路径正确 %.1f%%; 输出 %d 人, 对上金标 %d, 其中路径正确 %.1f%%" % (
         ok / n * 100, ok, n, fd / n * 100, ok / max(1, fd) * 100, tot["pred"], tot["matched"], tot["matched_ok"] / max(1, tot["matched"]) * 100),
     "按跳数: " + "; ".join("%s跳 %.1f%% (%d/%d, 找到 %.1f%%)" % (L if L < 4 else "4+", c["ok"] / c["n"] * 100, c["ok"], c["n"], c["found"] / c["n"] * 100) for L, c in sorted(by.items())),
     "错误 top (金标→预测): " + ", ".join("%s→%s×%d" % (g, p or "本人", k) for (g, p), k in errs.most_common(12))]
S.append("不分性别时的召回: %.1f%% (%d/%d)" % (tot["ok_neutral"] / n * 100, tot["ok_neutral"], n))
if a.vote: S.append("换根投票: %s" % dict(vstat))
open(os.path.join(OUT, "eval_%s.txt" % split), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
