# -*- coding: utf-8 -*-
"""
零标注起步: 用零样本大模型 (API) 给无标注墓志打"多视角伪标签"。
  直接视角: 以墓主为中心; 其他视角: 以直接结果里的 父/配偶/第一个子/祖父 为中心各问一次。
  用亲属代数检验各视角是否吻合 → support / conflict (与 pseudo_label.py 同格式), 供后面挑选训练样本。
输出: runs/<out>.jsonl   每行 {"id","root","g_root","prompt","direct","support","conflict","alt","views","n_views","weak"}
用法: python api_pseudo.py --data ../M_模型/sft/unlabeled.jsonl --out api_u2000 --limit 2000
可随时中断重跑: 已完成的文档会从输出文件读回, 不重复请求。
"""
import os, sys, json, time, argparse, threading, collections, concurrent.futures as cf
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
from kin_algebra import normalize, inverse, BASE, GENDER_OF, view_consistent, view_to_ego
from openai import OpenAI
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
MODEL = os.environ.get("LLM_MODEL", "openai-gpt-oss-120b")
client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1"), timeout=180, max_retries=1)
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True); OUT = os.path.join(HERE, "runs", a.out + ".jsonl")
_lock = threading.Lock(); _last = [0.0]


def ask(prompt):
    err = ""
    for k in range(5):
        with _lock:
            w = _last[0] + 1.1 - time.time()
            if w > 0: time.sleep(w)
            _last[0] = time.time()
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
            if r.choices[0].message.content: return r.choices[0].message.content
        except Exception as e:
            err = str(e); time.sleep(5 * (k + 1))
    return ""


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


def parse_G(out, g):
    """兼容 Markdown 表格 (| 姓名 | 关系码 |) 与普通 "姓名 | 关系码" 行"""
    res = {}
    for ln in out.splitlines():
        ln = ln.strip().strip("|").replace("**", "")
        p = [x.strip() for x in ln.split("|")]
        if len(p) == 2 and p[1] and set(p[1]) <= BASE and p[0] and p[0] not in res and p[0] not in ("姓名",): res[p[0]] = normalize(p[1], g)
    return res


data = [json.loads(l) for l in open(os.path.join(HERE, a.data), encoding="utf-8")]
if a.limit: data = data[:a.limit]
done = {}
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        if r["direct"]: done[r["id"]] = r
todo = [d for d in data if d["id"] not in done]
print("文档 %d, 已完成 %d, 待处理 %d" % (len(data), len(done), len(todo)), flush=True)
swap = lambda d, name: d["prompt"][:d["prompt"].index("「")] + "「" + name + d["prompt"][d["prompt"].index("」"):]


def work(d):
    g_ego = d["g_root"]; raw0 = ask(d["prompt"]); direct = parse_G(raw0, g_ego); raws = [raw0]
    support, conflict, alt, views = collections.Counter(), collections.Counter(), collections.defaultdict(list), []
    used = set()
    for name, path in direct.items():
        if path in ("F", "W", "H", "S", "FF") and path not in used and len(name) >= 2 and not name.endswith("氏"):
            used.add(path); g_y = GENDER_OF[path[-1]]
            rawv = ask(swap(d, name)); raws.append(rawv); view = parse_G(rawv, g_y); views.append([name, path, view])
            for n, p in view.items():
                if same(n, d["root"]) or same(d["root"], n):
                    (support if view_consistent(path, "", p, g_ego, g_y) else conflict)[name] += 1; continue
                key = next((k for k in direct if same(n, k) or same(k, n)), None)
                if key is None: alt[n] += view_to_ego(path, p, g_ego, g_y)[:1]
                elif view_consistent(path, direct[key], p, g_ego, g_y): support[key] += 1
                else: conflict[key] += 1; alt[key] += view_to_ego(path, p, g_ego, g_y)[:1]
    return dict(id=d["id"], root=d["root"], g_root=g_ego, prompt=d["prompt"], direct=direct, support=dict(support), conflict=dict(conflict), alt=dict(alt), views=views, n_views=len(views), weak=d.get("weak", {}), gold=d.get("gold", []), raw=raws)


n = 0
with open(OUT, "w", encoding="utf-8") as f0:
    for r in done.values(): f0.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.threads) as ex:
    for r in ex.map(work, todo):
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1
        if n % 25 == 0: print("done %d/%d" % (n, len(todo)), flush=True)
print("API_PSEUDO_DONE")
