# -*- coding: utf-8 -*-
"""
多视角解析 + 换根一致性: 给无标注 (或任意) 文档打伪标签, 并记录每个人被几个视角印证 / 否定。
  直接视角: 以墓主为中心解析一次
  其他视角: 以直接结果中的 父 / 配偶 / 第一个子 / 祖父 为中心各解析一次, 用代数换算回墓主视角
  对每个人: support = 与直接结果一致的视角数, conflict = 给出不同路径的视角数
用法: python pseudo_label.py --name GR --data sft/unlabeled.jsonl --out pseudo_unl [--limit 3000] [--shard 0/2]
输出: runs/<name>/<out>.jsonl  每行 {"id","root","g_root","prompt","direct":{名:路径},"support":{名:n},"conflict":{名:n},"n_views","weak","gold"}
"""
import os, sys, json, argparse, collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
from kin_algebra import normalize, inverse, BASE, GENDER_OF, view_consistent, view_to_ego

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default="0/1"); ap.add_argument("--bs", type=int, default=12)
a = ap.parse_args()
OUT = os.path.join(HERE, "runs", a.name)
data = [json.loads(l) for l in open(os.path.join(HERE, a.data), encoding="utf-8")]
if a.limit: data = data[:a.limit]
si, sn = map(int, a.shard.split("/")); data = data[si::sn]
tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0}), OUT).eval()


@torch.no_grad()
def generate(prompts):
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i])); res = {}
    for b in range(0, len(order), a.bs):
        idx = order[b:b + a.bs]
        txt = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True) for i in idx]
        x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        y = model.generate(**x, max_new_tokens=400, do_sample=False)
        for i, o in zip(idx, y): res[i] = tok.decode(o[x["input_ids"].shape[1]:], skip_special_tokens=True)
        if (b // a.bs) % 20 == 0: print("gen %d/%d" % (b, len(order)), flush=True)
    return [res[i] for i in range(len(prompts))]


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


def parse_G(out, g):
    res = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) == 2 and p[1] and set(p[1]) <= BASE and p[0] not in res: res[p[0]] = normalize(p[1], g)
    return res


direct = [parse_G(o, d["g_root"]) for o, d in zip(generate([d["prompt"] for d in data]), data)]
jobs = []
for di, (d, pr) in enumerate(zip(data, direct)):
    used = set()
    for name, path in pr.items():
        if path in ("F", "W", "H", "S", "FF") and path not in used and len(name) >= 2 and not name.endswith("氏"):
            used.add(path); jobs.append((di, name, path))
swap = lambda d, name: d["prompt"][:d["prompt"].index("「")] + "「" + name + d["prompt"][d["prompt"].index("」"):]
v_outs = generate([swap(data[di], name) for di, name, _ in jobs])
support = [collections.Counter() for _ in data]; conflict = [collections.Counter() for _ in data]; nviews = collections.Counter()
alt = [collections.defaultdict(list) for _ in data]; views = [[] for _ in data]
for (di, yname, ypath), o in zip(jobs, v_outs):
    d = data[di]; g_ego = d["g_root"]; g_y = GENDER_OF[ypath[-1]]; view = parse_G(o, g_y); nviews[di] += 1; views[di].append([yname, ypath, view])
    for n, p in view.items():
        if same(n, d["root"]) or same(d["root"], n):                             # 视角里看到的墓主: 检验 墓主→y 这条边本身
            (support if view_consistent(ypath, "", p, g_ego, g_y) or (g_ego is None and len(p) == len(ypath)) else conflict)[di][yname] += 1
            continue
        key = next((k for k in direct[di] if same(n, k) or same(k, n)), None)
        if key is None: alt[di][n] += view_to_ego(ypath, p, g_ego, g_y)[:1]      # 直接视角没列出的人: 记下换算回墓主视角的首选读法
        elif view_consistent(ypath, direct[di][key], p, g_ego, g_y): support[di][key] += 1
        else: conflict[di][key] += 1; alt[di][key] += view_to_ego(ypath, p, g_ego, g_y)[:1]
with open(os.path.join(OUT, "%s%s.jsonl" % (a.out, "" if sn == 1 else "_%d" % si)), "w", encoding="utf-8") as f:
    for di, d in enumerate(data):
        f.write(json.dumps(dict(id=d["id"], root=d["root"], g_root=d["g_root"], prompt=d["prompt"], direct=direct[di], support=dict(support[di]), conflict=dict(conflict[di]),
                                alt={k: v for k, v in alt[di].items()}, views=views[di], n_views=nviews[di], weak=d.get("weak", {}), gold=d.get("gold", [])), ensure_ascii=False) + "\n")
print("PSEUDO_DONE docs %d views %d" % (len(data), len(jobs)))
