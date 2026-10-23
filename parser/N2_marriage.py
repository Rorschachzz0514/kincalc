# -*- coding: utf-8 -*-
"""
N2 —— 网络应用示例: 从 N1 的语料级网络中抽出"婚姻边", 回答一个唐史问题:
       高门 (山东五姓 崔卢李郑王 / 关中郡姓 韦裴柳薛杨杜) 的通婚是否内卷 (endogamy), 并随唐初→晚唐怎样变化。
       同时比较: 只用标注墓志 (人工库补全后) vs. 加上无标注墓志的 A 级推断边 vs. 全部推断边, 结论是否一致。
婚姻边: 路径以 W/H/M 结尾的人物. 丈夫方姓氏 = 墓主姓 (当丈夫位于父系路径 {F,S,B}* 上) 或妻方人物本人姓;
        妻方姓氏 = 人物名的姓 (X氏 也算). 只保留双方姓氏都能确定的婚姻.
输出: runs/network/N2_marriage.json, N2_stats.txt
"""
import os, sys, json, re, collections, math
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); OUT = os.path.join(HERE, "runs", "network")
generic = lambda n: len(n) < 2 or n.endswith("某") or "□" in n or n.endswith("公") or n.endswith("君") or n.endswith("夫人")
rows = lambda t: [tuple(x.split(" | ")) for x in t.splitlines() if " | " in x]
DOUBLE = {"独孤", "长孙", "宇文", "尉迟", "令狐", "司马", "诸葛", "欧阳", "皇甫", "上官", "慕容", "贺兰", "豆卢", "拓跋", "叱干", "呼延", "万俟", "赫连", "库狄", "拓拔", "斛律", "叱罗", "夏侯", "公孙", "闾丘", "鲜于", "南宫", "东方", "端木", "轩辕", "太史", "淳于", "单于", "吐谷", "纥干", "乙速", "斛斯", "乞伏", "侯莫", "若干", "步六", "达奚", "叱列", "贺拔", "可朱", "车非", "屈突", "契苾", "阿史", "执失", "康阿", "安延"}
def surname(n):
    n = re.sub(r"[\s]", "", n)
    if len(n) < 2: return None
    if n[:2] in DOUBLE: return n[:2]
    if n[0] in "某□无不": return None
    return n[0]

# ---------- 载入所有文档 (同 N1)
docs = []
for l in open(os.path.join(HERE, "sft", "Aall_train.jsonl"), encoding="utf-8"):
    d = json.loads(l); docs.append(dict(id=d["id"], root=d["root"], src="labelled", people=[(n, p, "L") for n, p in rows(d["target"])]))
roots = {}
for sp in ("dev", "test"):
    for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % sp), encoding="utf-8"):
        d = json.loads(l); roots[d["id"]] = d["root"]
    for l in open(os.path.join(HERE, "runs", "Aall", "pred_%s.jsonl" % sp), encoding="utf-8"):
        d = json.loads(l); docs.append(dict(id=d["id"], root=roots[d["id"]], src="predicted", people=[(n, p, "P") for n, p in d["pred"].items()]))
for l in open(os.path.join(HERE, "runs", "GR", "pseudo_unl.jsonl"), encoding="utf-8"):
    d = json.loads(l); ppl = []
    for n, p in d["direct"].items():
        s, c = d["support"].get(n, 0), d["conflict"].get(n, 0)
        ppl.append((n, p, "D" if c else "A" if s >= 2 else "B" if s == 1 else "C"))
    docs.append(dict(id=d["id"], root=d["root"], src="unlabelled", people=ppl))

# ---------- 墓主元数据 (行号 = id 中的数字)
df = pd.read_excel(os.path.join(ROOT, "data", "external", "0_墓志数据", "tang_0915_epitaphs.xlsx"))
rid_of = lambda i: int(i[1:]) if i.startswith("U") else int(i.rsplit("_", 1)[1])
PERIODS = ["初唐", "盛唐", "中唐", "晚唐"]
def period(y, p):
    try: y = float(y)
    except Exception: y = float("nan")
    if y == y: return "初唐" if y < 713 else "盛唐" if y < 766 else "中唐" if y < 836 else "晚唐"
    return p if p in PERIODS else None

# ---------- 抽婚姻边
SHANDONG = set("崔卢李郑王"); GUANZHONG = set("韦裴柳薛杨杜"); ELITE = SHANDONG | GUANZHONG
patri = lambda q: all(ch in "FSB" for ch in q)
marr = []                                                      # (husband_surname, wife_surname, period, src, tier)
n_cand = collections.Counter()
for d in docs:
    r = df.iloc[rid_of(d["id"])]; xs = re.sub(r"\s", "", str(r["姓"])); ego_sn = xs if (1 <= len(xs) <= 2 and xs not in ("nan", "无记载", "无", "不详")) else surname(d["root"])
    per = period(r["去世年"], r["时期"]); g = str(r["性别"])
    if not ego_sn or not per: continue
    for n, p, t in d["people"]:
        if not p or generic(n) and not n.endswith("氏"): continue
        sn = surname(n)
        if not sn: continue
        h = w = None
        if p == "M": h, w = ego_sn, sn
        elif p == "W": h, w = ego_sn, sn
        elif p == "H": h, w = sn, ego_sn
        elif p[-1] in "WM" and patri(p[:-1]): h, w = ego_sn, sn           # 父系男性的妻/母
        elif p[-1] == "H" and len(p) >= 2 and p[-2] in "DZ" and patri(p[:-2]): h, w = sn, ego_sn   # 父系女性 (女/姊妹/孙女...) 的夫
        else: n_cand["other"] += 1; continue
        if h == w: n_cand["same_surname"] += 1; continue           # 同姓婚: 多为抽取错误或姓氏遗漏, 剔除
        marr.append((h, w, per, d["src"], t))
n_marr = len(marr); print("婚姻边", n_marr, "(其它路径", n_cand["other"], ", 同姓剔除", n_cand["same_surname"], ")")

# ---------- 统计: 高门内婚率 (观察 / 期望) 按时期 × 数据来源
def stats(sub):
    """sub: list of (h, w). 返回 内婚率, 期望内婚率(按边缘频率随机配对), 比值"""
    if len(sub) < 20: return None
    both = sum(1 for h, w in sub if h in ELITE and w in ELITE); ph = sum(1 for h, w in sub if h in ELITE) / len(sub); pw = sum(1 for h, w in sub if w in ELITE) / len(sub)
    obs = both / len(sub); exp = ph * pw
    return dict(n=len(sub), elite_both=round(obs * 100, 1), expected=round(exp * 100, 1), ratio=round(obs / exp, 2) if exp else None, elite_h=round(ph * 100, 1), elite_w=round(pw * 100, 1))
SOURCES = {"labelled": lambda m: m[3] in ("labelled", "predicted"), "unlabelled_A": lambda m: m[3] == "unlabelled" and m[4] == "A",
           "unlabelled_AB": lambda m: m[3] == "unlabelled" and m[4] in "AB", "unlabelled_all": lambda m: m[3] == "unlabelled", "all_A": lambda m: m[3] != "unlabelled" or m[4] == "A"}
res = {"n_marriages": n_marr, "by_source": {}, "by_period": {}, "periods": PERIODS}
for s, f in SOURCES.items():
    sub = [m for m in marr if f(m)]; res["by_source"][s] = dict(n=len(sub), overall=stats([(h, w) for h, w, *_ in sub]))
    res["by_period"][s] = {p: stats([(h, w) for h, w, pp, *_ in sub if pp == p]) for p in PERIODS}
# 山东五姓 × 关中郡姓 的互婚倾向
def block(sub, A, B):
    if not sub: return None
    hA = sum(1 for h, w in sub if h in A) / len(sub); wB = sum(1 for h, w in sub if w in B) / len(sub); ob = sum(1 for h, w in sub if h in A and w in B) / len(sub)
    return round(ob / (hA * wB), 2) if hA * wB > 0 else None
allA = [(h, w) for h, w, p, s, t in marr if s != "unlabelled" or t == "A"]
res["blocks_all_A"] = {"SD-SD": block(allA, SHANDONG, SHANDONG), "SD-GZ": block(allA, SHANDONG, GUANZHONG), "GZ-SD": block(allA, GUANZHONG, SHANDONG), "GZ-GZ": block(allA, GUANZHONG, GUANZHONG),
                       "elite-other": block(allA, ELITE, set(w for h, w in allA) - ELITE), "other-elite": block(allA, set(h for h, w in allA) - ELITE, ELITE)}
# 姓氏 × 姓氏 观察/期望 (对称化: 不分夫妻方向), 取婚姻数最多的 12 个姓
cnt = collections.Counter(); deg = collections.Counter()
for h, w in allA: cnt[(h, w)] += 1; cnt[(w, h)] += 1; deg[h] += 1; deg[w] += 1
top = [s for s, _ in deg.most_common(12)]; N2 = 2 * len(allA)
mat = [[None] * len(top) for _ in top]; raw = [[0] * len(top) for _ in top]
for i, a in enumerate(top):
    for j, b in enumerate(top):
        exp = deg[a] * deg[b] / N2; raw[i][j] = cnt[(a, b)]; mat[i][j] = round(math.log2((cnt[(a, b)] + 0.5) / (exp + 0.5)), 2)
res["surname_matrix"] = {"surnames": top, "log2_obs_over_exp": mat, "counts": raw, "n_marriages": len(allA)}
# 新增知识: 无标注墓志贡献的婚姻边 (A 级) 占比; 与人工库原标签的比较 (标注集里补全前后)
g_marr = 0
for l in open(os.path.join(HERE, "sft", "G_train.jsonl"), encoding="utf-8"):
    d = json.loads(l)
    for n, p in rows(d["target"]):
        if p and (p[-1] in "WHM"): g_marr += 1
res["labelled_marriage_paths_original_labels"] = g_marr; res["labelled_marriage_paths_completed"] = sum(1 for m in marr if m[3] == "labelled")
# 跨篇: 最长父系链 (基本边 F) 与跨篇家族
json.dump(res, open(os.path.join(OUT, "N2_marriage.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
S = ["N2 婚姻网络应用", "婚姻边 %d (标注+预测 %d, 无标注 %d; 其中 A 级 %d)" % (n_marr, sum(m[3] != "unlabelled" for m in marr), sum(m[3] == "unlabelled" for m in marr), sum(m[3] == "unlabelled" and m[4] == "A" for m in marr))]
for s in SOURCES: S.append("%-14s 总体 %s" % (s, res["by_source"][s]["overall"])); S += ["    %s %s" % (p, res["by_period"][s][p]) for p in PERIODS]
S.append("block ratios (all+A): %s" % res["blocks_all_A"]); S.append("top surnames: %s" % top)
S.append("标注集原标签婚姻路径 %d → 补全后 %d" % (g_marr, res["labelled_marriage_paths_completed"]))
open(os.path.join(OUT, "N2_stats.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
