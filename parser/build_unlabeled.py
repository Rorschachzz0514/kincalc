# -*- coding: utf-8 -*-
"""
无标注池: 原始墓志表 (10,201 行) 中不在 EpiNet 标注集里的行, 每行一个墓主。
输出 sft/unlabeled.jsonl: {"id","root","g_root","prompt","weak":{"F":父名,"FF":祖父名}}
  weak = 我方表里的 父亲/祖父 列 (此前由大模型抽取并经人工抽查, 与 CBDB 一致率约 86%), 只用来粗略检验伪标签质量, 不参与训练。
"""
import os, re, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, zhconv
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "训练集", "epinet_v0"); DST = os.path.join(HERE, "sft")
t2s = lambda s: zhconv.convert(str(s), "zh-cn")

labeled = set()
for sp in ("train", "dev", "test"):
    for l in open(os.path.join(SRC, sp + ".jsonl"), encoding="utf-8"):
        d = json.loads(l)
        labeled.add(d["rid"]); labeled.update(s["rid"] for s in d["subjects"])
tpl = json.loads(open(os.path.join(DST, "G_train.jsonl"), encoding="utf-8").readline())["prompt"]
head = tpl[:tpl.index("「")]; mid = tpl[tpl.index("」"):tpl.index("墓志:\n") + 4]

df = pd.read_excel(os.path.join(ROOT, "data", "external", "0_墓志数据", "tang_0915_epitaphs.xlsx"))
cols = list(df.columns); iN, iG, iT, iF, iFF = cols.index("姓名"), cols.index("性别"), cols.index("原文"), cols.index("父亲"), cols.index("祖父")
n = 0; skip = 0
with open(os.path.join(DST, "unlabeled.jsonl"), "w", encoding="utf-8") as f:
    for rid in range(len(df)):
        if rid in labeled: continue
        r = df.iloc[rid]; name = t2s(re.sub(r"\s", "", str(r.iloc[iN]))); raw = str(r.iloc[iT])
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if lines and len(lines[0]) <= 12: lines = lines[1:]                      # 首行是汇编编号
        text = t2s(re.sub(r"\s+", "", "".join(lines)))
        if len(name) < 2 or name in ("nan", "无记载") or len(text) < 150: skip += 1; continue
        if len(text) > 2400: text = text[:1600] + "……" + text[-800:]
        weak = {}
        for k, i in (("F", iF), ("FF", iFF)):
            v = t2s(re.sub(r"\s", "", str(r.iloc[i])))
            if 2 <= len(v) <= 4 and v not in ("nan", "无记载", "不详", "未知"): weak[k] = v
        g = {"男": 1, "女": 0}.get(str(r.iloc[iG]).strip())
        f.write(json.dumps(dict(id="U%d" % rid, root=name, g_root=g, gold=[], maxhop=0, weak=weak, prompt=head + "「" + name + mid + text), ensure_ascii=False) + "\n"); n += 1
print("无标注文档 %d (跳过无名/过短 %d); 已标注行 %d" % (n, skip, len(labeled)))
