# -*- coding: utf-8 -*-
"""
L1 —— 跨篇"同一人"判定: 换根对齐是否比只看名字强?  (先用金标家谱树, 检验方法本身)
候选对: 不同墓志里姓名完全相同的两个人 (去掉 X氏 这类无名者)。金标: CBDB 人物 id 是否相同。
方法:
  name     只要同名就判同一人
  ego-feat 不换根: 只比较两人各自相对"本篇墓主"的路径是否兼容 —— 没有代数时能做的最好的事
  reroot   把两篇的家谱都换根到这个候选人身上, 比较同一位置上的亲属姓名:
             唯一位置 (父/母/祖父/夫...) 同名 → 支持, 异名 → 否定; 可多人的位置 (子/女/妻/兄弟) 有交集 → 支持
           判定: 有支持且无否定 → 同一人; 有否定 → 不同人; 无任何可比位置 → 弃权 (回退到 name)
输出: runs/L1_summary.txt
用法: python L1_crossdoc_link.py [gold | <pred 文件前缀>]
"""
import os, sys, json, collections, itertools
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
from kin_algebra import core, normalize, reroot, GENDER_OF
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "训练集", "epinet_v0")
UNIQ = set("FMH")
generic = lambda n: len(n) < 2 or n.endswith("氏") or n.endswith("某") or "□" in n

docs = []
for sp in ("train", "dev", "test"):
    for line in open(os.path.join(SRC, sp + ".jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["ego"] is None: continue
        ego = next(n for n in d["nodes"] if n["id"] == d["ego"]); g = ego["gender"]
        people = {}                                                       # id -> (name, path, gender)
        for n in d["nodes"]:
            c = "" if n["id"] == d["ego"] else core(n.get("path_from_ego"))
            if c is None or not n["in_doc"]: continue
            people[n["id"]] = (n["name_simp"], normalize(c, g), n["gender"])
        docs.append(dict(id=d["id"], g=g, people=people))


def view(doc, pid):
    """以 pid 为中心: {路径: {姓名,...}}"""
    name, p, gx = doc["people"][pid]; out = collections.defaultdict(set)
    for qid, (qn, qp, qg) in doc["people"].items():
        if qid == pid or generic(qn): continue
        r = reroot(p, qp, doc["g"], gx)
        if r: out[r].add(qn)
    return out


def compare(va, vb):
    sup = con = 0
    for r in set(va) & set(vb):
        if all(ch in UNIQ for ch in r):
            if va[r] & vb[r]: sup += 1
            else: con += 1
        elif va[r] & vb[r]: sup += 1
    return sup, con


by_name = collections.defaultdict(list)
for di, d in enumerate(docs):
    for pid, (n, p, g) in d["people"].items():
        if not generic(n): by_name[n].append((di, pid))
pairs = []
for n, ms in by_name.items():
    for (da, pa), (db, pb) in itertools.combinations(ms, 2):
        if da != db: pairs.append((n, da, pa, db, pb))
pos = sum(1 for _, da, pa, db, pb in pairs if pa == pb)
st = collections.Counter(); ex = []
for n, da, pa, db, pb in pairs:
    y = pa == pb
    sup, con = compare(view(docs[da], pa), view(docs[db], pb))
    dec = "同" if sup and not con else "异" if con else "弃权"
    st[(dec, y)] += 1
    # 不换根的特征: 两人相对各自墓主的路径相同 (如都是"父") 说明不了什么; 只能比较"都是墓主"的情形
    ea = docs[da]["people"][pa][1] == ""; eb = docs[db]["people"][pb][1] == ""
    st[("ego_both" if ea and eb else "ego_one" if ea or eb else "ego_none", y)] += 1
    na = {v[0] for k, v in docs[da]["people"].items() if k != pa and not generic(v[0])}; nb = {v[0] for k, v in docs[db]["people"].items() if k != pb and not generic(v[0])}
    ov = len((na & nb) - {n}); st[("bag%d" % min(ov, 2), y)] += 1
    st[("bag_veto", y)] += ov >= 1 and con == 0
    st[("bag_or_sup", y)] += (ov >= 1 and con == 0) or (sup >= 1 and con == 0)
    st[("bag_veto2", y)] += ov >= 1 and not (con >= 1 and sup == 0)
    st[("lenient", y)] += (sup >= 1 and con <= sup - 1) or (sup >= 1 and con == 0)
    if dec == "同" and not y and len(ex) < 6: ex.append((n, docs[da]["id"], docs[db]["id"], sup, con))

def prf(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); return "P %.1f%% R %.1f%% F1 %.1f%%" % (p * 100, r * 100, 2 * p * r / max(1e-9, p + r) * 100)
T = lambda k: st[(k, True)]; F = lambda k: st[(k, False)]
S = ["L1 跨篇同一人判定 (金标家谱树, %d 篇)" % len(docs),
     "同名候选对 %d; 其中确为同一人 %d (%.1f%%) —— 即'同名就算同一人'的精确率" % (len(pairs), pos, pos / len(pairs) * 100),
     "换根对齐的三种结论:  判同 %d 对 (其中真同 %d, %.1f%%);  判异 %d 对 (其中真异 %d, %.1f%%);  弃权 %d 对 (其中真同 %d, %.1f%%)" % (
         T("同") + F("同"), T("同"), T("同") / max(1, T("同") + F("同")) * 100, T("异") + F("异"), F("异"), F("异") / max(1, T("异") + F("异")) * 100,
         T("弃权") + F("弃权"), T("弃权"), T("弃权") / max(1, T("弃权") + F("弃权")) * 100),
     "方法对比 (判'同一人'的 P/R/F1):",
     "  只看名字:                 " + prf(pos, len(pairs) - pos, 0),
     "  换根对齐 (只取判同):      " + prf(T("同"), F("同"), T("异") + T("弃权")),
     "  换根对齐 + 弃权回退名字:  " + prf(T("同") + T("弃权"), F("同") + F("弃权"), T("异")),
     "  亲属名单重叠>=1 (不看位置): " + prf(T("bag1") + T("bag2"), F("bag1") + F("bag2"), T("bag0")),
     "  亲属名单重叠>=2 (不看位置): " + prf(T("bag2"), F("bag2"), T("bag0") + T("bag1")),
     "  换根对齐 (支持数>否定数):   " + prf(T("lenient"), F("lenient"), pos - T("lenient")),
     "  名单重叠>=1 且换根无否定:    " + prf(T("bag_veto"), F("bag_veto"), pos - T("bag_veto")),
     "  名单重叠>=1 且非(只有否定):  " + prf(T("bag_veto2"), F("bag_veto2"), pos - T("bag_veto2")),
     "  (名单重叠 或 换根支持) 且无否定: " + prf(T("bag_or_sup"), F("bag_or_sup"), pos - T("bag_or_sup")),
     "不换根时能用的信息很少: 两人都是各自墓主 %d 对 (真同 %d); 一方是墓主 %d 对 (真同 %d); 都不是墓主 %d 对 (真同 %d)" % (
         T("ego_both") + F("ego_both"), T("ego_both"), T("ego_one") + F("ego_one"), T("ego_one"), T("ego_none") + F("ego_none"), T("ego_none")),
     "判同但库中为不同 id 的样例 (可能是库中重复录入): %s" % ex]
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
open(os.path.join(HERE, "runs", "L1_summary.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
