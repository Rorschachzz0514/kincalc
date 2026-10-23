# -*- coding: utf-8 -*-
"""
C1 —— "换根一致性"作为可信度信号, 比常见的无标注可信度信号强吗?
对交叉拟合模型在"没见过的那一半训练集"上的每条预测 (能对上金标的人), 计算三种可信度:
  logprob   生成该行时的平均 token 对数概率 (以及最小值)
  sample    采样自一致性: 温度 0.7 采样 5 次, 有几次给出同名同路径
  reroot    换根一致性: 其他视角印证数 / 否定数 (来自 xfit.jsonl, 已算好)
然后比较它们区分"路径对/错"的能力 (AUROC, 以及取最可信的 50% / 70% / 90% 时的正确率)。
用法: python conf_baselines.py run --name X0 --data runs/X0/xfit.jsonl     (GPU)
      python conf_baselines.py report runs/X0/conf.jsonl runs/X1/conf.jsonl
"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
sys.stdout.reconfigure(encoding="utf-8")


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


if sys.argv[1] == "run":
    import argparse, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from kin_algebra import normalize, BASE
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--name", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(); OUT = os.path.join(HERE, "runs", a.name)
    data = [json.loads(l) for l in open(os.path.join(HERE, a.data), encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0}), OUT).eval()

    def parse_line(ln, g):
        p = [x.strip() for x in ln.split("|")]
        return (p[0], normalize(p[1], g)) if len(p) == 2 and p[1] and set(p[1]) <= BASE else None

    order = sorted(range(len(data)), key=lambda i: -len(data[i]["prompt"])); rec = {}
    with torch.no_grad():
        for b in range(0, len(order), a.bs):
            idx = order[b:b + a.bs]
            txt = [tok.apply_chat_template([{"role": "user", "content": data[i]["prompt"]}], tokenize=False, add_generation_prompt=True) for i in idx]
            x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda"); L = x["input_ids"].shape[1]
            g = model.generate(**x, max_new_tokens=400, do_sample=False, output_scores=True, return_dict_in_generate=True)
            ts = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
            s = model.generate(**x, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=a.k)
            for j, i in enumerate(idx):
                gr = data[i]["g_root"]; lines = {}; cur = []; buf = ""
                for t, lp in zip(g.sequences[j, L:].tolist(), ts[j].tolist()):
                    if t in (tok.eos_token_id, tok.pad_token_id) or tok.decode([t]) in ("<|im_end|>", "<|endoftext|>"): break
                    piece = tok.decode([t]); buf += piece; cur.append(lp)
                    if "\n" in piece:
                        r = parse_line(buf.strip(), gr)
                        if r and r[0] not in lines: lines[r[0]] = (r[1], sum(cur) / len(cur), min(cur))
                        buf = ""; cur = []
                if buf.strip() and cur:
                    r = parse_line(buf.strip(), gr)
                    if r and r[0] not in lines: lines[r[0]] = (r[1], sum(cur) / len(cur), min(cur))
                samples = []
                for q in range(a.k):
                    o = tok.decode(s[j * a.k + q, L:], skip_special_tokens=True)
                    samples.append(dict(filter(None, (parse_line(ln, gr) for ln in o.splitlines()))))
                rec[i] = (lines, samples)
            if (b // a.bs) % 10 == 0: print("conf %d/%d" % (b, len(order)), flush=True)
    with open(os.path.join(OUT, "conf.jsonl"), "w", encoding="utf-8") as f:
        for i, d in enumerate(data):
            lines, samples = rec[i]
            for n, (p, lpm, lpmin) in lines.items():
                gm = [x["path"] for x in d["gold"] if same(n, x["name"])]
                if not gm: continue
                sc = sum(1 for sm in samples if any((m == n or same(m, n) or same(n, m)) and q == p for m, q in sm.items()))
                f.write(json.dumps(dict(id=d["id"], name=n, path=p, ok=p in gm, lp_mean=lpm, lp_min=lpmin, sample=sc,
                                        sup=d["support"].get(n, 0) if d["direct"].get(n) == p else 0, con=d["conflict"].get(n, 0) if d["direct"].get(n) == p else 0,
                                        covered=d["direct"].get(n) == p), ensure_ascii=False) + "\n")
    print("CONF_DONE")
else:
    R = [json.loads(l) for f in sys.argv[2:] for l in open(os.path.join(HERE, f), encoding="utf-8")]
    R = [r for r in R if r["covered"]]

    def auroc(score):
        pos = [score(r) for r in R if r["ok"]]; neg = [score(r) for r in R if not r["ok"]]
        allv = sorted(set(pos + neg)); rank = {}
        cnt = collections.Counter(pos + neg); c = 0
        for v in allv: rank[v] = c + (cnt[v] + 1) / 2; c += cnt[v]
        return (sum(rank[v] for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    def acc_at(score, cov):
        rs = sorted(R, key=lambda r: -score(r)); k = int(len(rs) * cov); return sum(r["ok"] for r in rs[:k]) / k

    reroot = lambda r: (r["sup"] if r["con"] == 0 else -r["con"])
    scores = collections.OrderedDict([("平均对数概率", lambda r: r["lp_mean"]), ("最小对数概率", lambda r: r["lp_min"]), ("采样自一致性 (5 次)", lambda r: r["sample"] + r["lp_mean"] * 1e-3),
                                      ("换根一致性", lambda r: reroot(r) + r["lp_mean"] * 1e-3), ("换根 + 采样", lambda r: reroot(r) + r["sample"] * 0.6 + r["lp_mean"] * 1e-3)])
    n = len(R); acc = sum(r["ok"] for r in R) / n
    S = ["C1 可信度信号对比 (交叉拟合模型在未见过的半个训练集上; 能对上金标的预测 %d 条, 总体路径正确率 %.1f%%, 错误 %d 条)" % (n, acc * 100, n - sum(r["ok"] for r in R)),
         "信号                   AUROC    最可信50%的正确率   70%     90%"]
    for k, f in scores.items():
        S.append("%-18s  %.3f     %.1f%%            %.1f%%   %.1f%%" % (k, auroc(f), acc_at(f, .5) * 100, acc_at(f, .7) * 100, acc_at(f, .9) * 100))
    open(os.path.join(HERE, "runs", "C1_conf_baselines.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
