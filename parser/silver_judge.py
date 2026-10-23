# -*- coding: utf-8 -*-
"""
银标评测 (pooling): 人工库标签不完整也有错, 只对着它算分会低估模型、也看不出"纠错"的效果。
做法与检索评测的 pooling 相同: 把金标 + 各个系统的预测合成候选池, 由一个独立的大模型 (与被评模型不同家族)
逐条对照原文裁决 对/错; 裁决为"对"的集合 = 银标。裁决结果按 (文档, 姓名, 路径) 缓存, 新系统只裁决新增候选。
注意: 这是模型裁决的"银标", 不是人工金标; 论文定稿前需人工复核 (本脚本会输出与人工抽查样本的一致率, 若提供)。
用法: python silver_judge.py --split dev --sys G=runs/G/pred_dev.jsonl X0=runs/X0/xdev.jsonl ...
输出: runs/judge_cache_<split>.json, runs/silver_<split>.txt
"""
import os, sys, json, time, argparse, threading, collections, concurrent.futures as cf
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--sys", nargs="+", default=[]); ap.add_argument("--nojudge", action="store_true")
a = ap.parse_args()
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-oss-120b")
client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1"), timeout=240, max_retries=1)
TAG = os.environ.get("JUDGE_TAG", "")                                 # 换裁决模型/提示词时用不同缓存
CACHE = os.path.join(HERE, "runs", "judge_cache_%s%s.json" % (a.split, TAG))
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

PROMPT = """你是唐代墓志与亲属称谓方面的专家。下面是一篇墓志和若干条候选判断, 每条说某个人相对墓主「%s」的亲属关系。请逐条对照原文裁决。
关系码从墓主出发逐步读: F=父 M=母 S=子 D=女 H=夫 W=妻 B=兄弟 Z=姊妹。例: FF=祖父, FFF=曾祖, FFFF=高祖, SS=孙, WF=岳父(妻之父), DH=女婿, HF=夫之父, BS=侄, FB=伯叔, MF=外祖父。
裁决方法: 先在原文中找到这个人的名 —— 墓志通常只写名不写姓 (如「祖亮」「嗣子岌」「曾祖讳河石」), 候选里的姓是按家族补上的, **姓不在原文里不算错**; 只要去掉姓之后的名在原文中作为人名出现, 就算找到。然后看原文对他的称谓 (祖/曾祖/考/嗣子/夫人之父/某某之孙…) 换算成的关系码是否与候选一致。
注意: 墓主是女性时, 「夫之祖」是 HFF 而不是 FF; 「X之孙」表示墓主是 X 的孙, 即 X 是墓主的 FF; 「大父/王父」= 祖父, 「曾王父/大王父」= 曾祖。
判「错」的情形: 关系码与原文称谓不符; 或这个名根本不是人名 (官职、地名、谥号、撰文书丹者); 或名字多字/少字。
每条输出一行: 编号 | 对 或 错 | 原文依据 (照抄含此人名的 4-12 个字) | 若错, 写正确的关系码或原因
只输出这些行。

墓志:
%s

候选:
%s"""
_lock = threading.Lock(); _last = [0.0]


def ask(prompt):
    err = ""
    for k in range(4):
        with _lock:
            w = _last[0] + 1.1 - time.time()
            if w > 0: time.sleep(w)
            _last[0] = time.time()
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=6000, messages=[{"role": "user", "content": prompt}])
            if r.choices[0].message.content: return r.choices[0].message.content
        except Exception as e:
            err = str(e); time.sleep(5 * (k + 1))
    return "ERROR " + err


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


docs = {}
for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % a.split), encoding="utf-8"):
    d = json.loads(l); docs[d["id"]] = d
systems = collections.OrderedDict(); systems["金标(CBDB)"] = {i: dict(tuple(x.split(" | ")) for x in d["target"].splitlines()) for i, d in docs.items()}
for s in a.sys:
    name, path = s.split("="); systems[name] = {}
    for l in open(os.path.join(HERE, path), encoding="utf-8"):
        r = json.loads(l); systems[name][r["id"]] = r.get("pred") or r.get("direct") or {}

todo = {}
for i, d in docs.items():
    cands = sorted({(n, p) for s in systems.values() for n, p in s.get(i, {}).items() if p})
    new = [c for c in cands if "%s\t%s\t%s" % (i, c[0], c[1]) not in cache]
    if new: todo[i] = new
if os.environ.get("ONLY_IDS"):                                        # 只裁决指定文档 (json 列表)
    keep = set(json.load(open(os.environ["ONLY_IDS"], encoding="utf-8")))
    todo = {i: c for i, c in todo.items() if i in keep}; docs = {i: d for i, d in docs.items() if i in keep}
if os.environ.get("ONLY_AUDIT"):                                      # 只裁决抽查样本所在的文档 (用来比较裁决模型)
    keep = {x[0] for x in json.load(open(os.path.join(HERE, "audit_claude_%s.json" % a.split), encoding="utf-8"))["items"]}
    todo = {i: c for i, c in todo.items() if i in keep}


def judge(i):
    d = docs[i]; text = d["prompt"][d["prompt"].index("墓志:\n") + 4:]; cs = todo[i]
    out = ask(PROMPT % (d["root"], text, "\n".join("%d. %s | %s" % (k + 1, n, p) for k, (n, p) in enumerate(cs))))
    res = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.replace("｜", "|").split("|")]
        if len(p) >= 2 and p[0].rstrip(".").isdigit():
            k = int(p[0].rstrip(".")) - 1
            if 0 <= k < len(cs) and p[1][:1] in "对错": res[k] = dict(ok=p[1][:1] == "对", note=" / ".join(p[2:]))
    return i, res, out


if todo and not a.nojudge:
    print("需要裁决的文档 %d, 候选 %d" % (len(todo), sum(map(len, todo.values()))), flush=True); n = 0
    with cf.ThreadPoolExecutor(3) as ex:
        for i, res, out in ex.map(judge, list(todo)):
            for k, v in res.items(): cache["%s\t%s\t%s" % (i, todo[i][k][0], todo[i][k][1])] = v
            n += 1
            if n % 20 == 0:
                json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False); print("judged %d/%d" % (n, len(todo)), flush=True)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)

# 银标: 每篇中被裁决为"对"的候选, 按人合并 (同名规则, 或 路径相同且一方的字是另一方的子集)
def merge(items):
    cl = []
    for n, p in items:
        for c in cl:
            if c["path"] == p and any(same(n, m) or same(m, n) or set(n) <= set(m) or set(m) <= set(n) for m in c["names"]): c["names"].add(n); break
        else: cl.append(dict(path=p, names={n}))
    return cl
silver = {}; unj = 0
for i in docs:
    ok = []
    for s in systems.values():
        for n, p in s.get(i, {}).items():
            v = cache.get("%s\t%s\t%s" % (i, n, p))
            if v is None: unj += 1
            elif v["ok"]: ok.append((n, p))
    silver[i] = merge(sorted(set(ok)))
tot = sum(len(v) for v in silver.values())
S = ["银标评测 (%s, %d 篇; 裁决模型 %s; 银标人物 %d; 未裁决候选 %d)" % (a.split, len(docs), MODEL, tot, unj), "系统             输出   判对   精确率   召回率   F1     (对 CBDB 金标的召回)"]
gold_eval = {i: d["gold"] for i, d in docs.items()}
for name, s in systems.items():
    n_out = n_ok = n_hit = g_ok = g_n = 0
    for i in docs:
        pr = s.get(i, {})
        for n, p in pr.items():
            if not p: continue
            n_out += 1; v = cache.get("%s\t%s\t%s" % (i, n, p)); n_ok += bool(v and v["ok"])
        for c in silver[i]: n_hit += any(p == c["path"] and n in c["names"] for n, p in pr.items())
        for g in gold_eval[i]: g_n += 1; g_ok += any(same(n, g["name"]) and p == g["path"] for n, p in pr.items())
    P = n_ok / max(1, n_out); R = n_hit / max(1, tot)
    S.append("%-14s %5d %6d   %5.1f%%   %5.1f%%   %5.1f%%   %5.1f%%" % (name, n_out, n_ok, P * 100, R * 100, 2 * P * R / max(1e-9, P + R) * 100, g_ok / max(1, g_n) * 100))
gs = systems["金标(CBDB)"]; bad = [(i, n, p, cache["%s\t%s\t%s" % (i, n, p)]["note"]) for i in docs for n, p in gs[i].items() if cache.get("%s\t%s\t%s" % (i, n, p), {"ok": True})["ok"] is False]
S.append("被裁决为错的金标条目 %d 条, 样例: %s" % (len(bad), "; ".join("%s %s|%s→%s" % b for b in bad[:15])))
AUD = os.path.join(HERE, "audit_claude_%s.json" % a.split)
if os.path.exists(AUD):                                              # 裁决模型与抽查样本的一致率
    its = json.load(open(AUD, encoding="utf-8"))["items"]; agree = n = 0; dis = []
    for i, nm, p, ok in its:
        v = cache.get("%s\t%s\t%s" % (i, nm, p))
        if v is None: continue
        n += 1; agree += v["ok"] == ok
        if v["ok"] != ok: dis.append("%s %s|%s 抽查=%s 裁决=%s" % (i, nm, p, "对" if ok else "错", "对" if v["ok"] else "错"))
    S.append("裁决模型与抽查样本 (Claude 逐条核对, 非人工金标) 的一致率: %d/%d (%.1f%%); 不一致: %s" % (agree, n, agree / max(1, n) * 100, "; ".join(dis)))
open(os.path.join(HERE, "runs", "silver_%s%s.txt" % (a.split, os.environ.get("OUT_TAG", ""))), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
