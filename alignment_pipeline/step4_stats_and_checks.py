# -*- coding: utf-8 -*-
"""
Step 4  统计 + 质量校验

  1. 训练集规模、路径分布、span 定位率
  2. 交叉校验: 认出 ego 的篇里, CBDB 给的 F / FF 与我方表的 父亲 / 祖父 列是否一致 (写法差异允许: 繁简、去姓)
  3. 抽一篇样本打印, 肉眼核对
输出  HUIBIAN_DIR/统计.txt
"""
import os, sys, json, collections, re
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from common import HUIBIAN_DIR, TRAIN_DIR, RAW_XLSX, t2t, clean_name


def same(a, b):
    a, b = t2t(clean_name(a)), t2t(clean_name(b))
    return a == b or (len(a) >= 2 and len(b) >= 2 and a[0] == b[0] and a[1:] == b[1:])


def main():
    S = {s: [json.loads(l) for l in open(os.path.join(TRAIN_DIR, s + ".jsonl"), encoding="utf-8")] for s in ("train", "dev", "test")}
    allx = sum(S.values(), [])
    e = pd.read_excel(RAW_XLSX); e = e.loc[:, ~e.columns.duplicated()]
    L = ["EpiNet v0 统计 (%s)" % pd.Timestamp.now().strftime("%Y-%m-%d")]
    L.append("篇数: train %d / dev %d / test %d = %d" % (len(S["train"]), len(S["dev"]), len(S["test"]), len(allx)))
    L.append("认出 ego: %d (%.1f%%); 合葬多墓主篇 %d" % (sum(x["ego"] is not None for x in allx), sum(x["ego"] is not None for x in allx) / len(allx) * 100, sum(len(x["subjects"]) > 1 for x in allx)))
    nd = [n for x in allx for n in x["nodes"]]; ind = [n for n in nd if n["in_doc"]]
    L.append("节点: 本篇登记 %d (span 定位 %.1f%%), 边外端 %d (定位 %.1f%%); 边 %d, 每篇 %.1f"
             % (len(ind), sum(n["span"] is not None for n in ind) / len(ind) * 100, len(nd) - len(ind),
                sum(n["span"] is not None for n in nd if not n["in_doc"]) / max(len(nd) - len(ind), 1) * 100, sum(len(x["edges"]) for x in allx), sum(len(x["edges"]) for x in allx) / len(allx)))
    pc = collections.Counter(n["path_from_ego"] for n in nd if n.get("path_from_ego") and n["path_from_ego"] != "EGO")
    L.append("有相对 ego 路径的节点 %d; 路径种类 %d; top40: %s" % (sum(pc.values()), len(pc), pc.most_common(40)))
    ec = collections.Counter(ed["rel"] for x in allx for ed in x["edges"]); L.append("边类型 %d 种; top30: %s" % (len(ec), ec.most_common(30)))
    # 交叉校验
    tot = agree = 0; dis = []
    for x in allx:
        if x["ego"] is None: continue
        row = e.iloc[x["subjects"][0]["rid"]]
        for col, code in (("父亲", "F"), ("祖父", "FF")):
            g = row[col]
            if pd.isna(g) or str(g) in ("无记载", "未知"): continue
            cb = [n["name"] for n in x["nodes"] if n.get("path_from_ego") == code]
            if not cb: continue
            tot += 1
            if any(same(g, y) for y in cb): agree += 1
            elif len(dis) < 10: dis.append((x["code"], col, str(g), cb))
    L.append("交叉校验 (我方表 父亲/祖父 vs CBDB 路径 F/FF): %d 对, 一致 %d (%.1f%%); 不一致示例: %s" % (tot, agree, agree / max(tot, 1) * 100, dis))
    # 样本
    x = max((y for y in S["train"] if y["ego"] and len(y["text"]) < 900), key=lambda y: sum(n["span"] is not None for n in y["nodes"]))
    L += ["", "---- 样本 %s (%s) ----" % (x["id"], x["title"]), x["text"][:300] + " ...", "nodes:"]
    L += ["  %-10s path=%-8s span=%-12s in_doc=%s %s" % (n["name"], n["path_from_ego"], n["span"], n["in_doc"], n["offices"][:2]) for n in x["nodes"]]
    open(os.path.join(HUIBIAN_DIR, "统计.txt"), "w", encoding="utf-8").write("\n".join(map(str, L)))
    print("\n".join(map(str, L[:8])))


if __name__ == "__main__":
    main()
