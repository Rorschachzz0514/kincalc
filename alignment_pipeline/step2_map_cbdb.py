# -*- coding: utf-8 -*-
"""
Step 2  把每篇《汇编》原文映射到 CBDB (正编 / 续集), 输出映射表和缺失清单

输入  汇编文本_分块.jsonl;  CBDB: BIOG_SOURCE_DATA (出处) + BIOG_MAIN (人物)
输出  HUIBIAN_DIR/汇编_CBDB映射.csv        每篇: code, rid, 墓主, book, cbdb_persons(命中人名数), matched
      HUIBIAN_DIR/CBDB有_我方缺_编号清单.csv  CBDB 录过但我方没有原文的编号 (正编/续集分列, 附人物名)

规则
  1. CBDB 出处页码 'Kaiyuan 383' → 开元383;  '(XJ)...' 表示续集。年号拼音→汉字用 NIAN_HAO 表
  2. 同一编号正编和续集都可能有 (两本书各自从 1 编号)。判定本篇属哪本书:
       分别取两本书在该编号下登记的人物, 数有多少人名出现在本篇原文里 (name_in_text), 多者胜; 都为 0 → 未映射
  3. matched=False 的两种情况: (a) CBDB 根本没录这一号; (b) 录了但人名一个都对不上 (多为异体字/校勘差异, 或撞号)
"""
import os, sys, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from common import HUIBIAN_DIR, cbdb, era_tables, code_from_cbdb_pages, name_in_text, t2s, TEXTID_ZHENGBIAN, TEXTID_XUJI


def load_cbdb_sources():
    pin, _ = era_tables()
    src = pd.read_sql(f"""select s.c_personid, s.c_textid, s.c_pages, b.c_name_chn
                          from BIOG_SOURCE_DATA s join BIOG_MAIN b on s.c_personid=b.c_personid
                          where s.c_textid in ({TEXTID_ZHENGBIAN},{TEXTID_XUJI})""", cbdb())
    src["code"] = src.c_pages.map(lambda p: code_from_cbdb_pages(p, pin))
    src["book"] = src.c_textid.map({TEXTID_ZHENGBIAN: "正编", TEXTID_XUJI: "续集"})
    return src.dropna(subset=["code"])


def main():
    docs = [json.loads(l) for l in open(os.path.join(HUIBIAN_DIR, "汇编文本_分块.jsonl"), encoding="utf-8")]
    src = load_cbdb_sources()
    by = {k: g for k, g in src.groupby(["book", "code"])}
    rows = []
    for d in docs:
        ts = t2s(d["text"]); best = (0, None)
        for book in ("正编", "续集"):
            g = by.get((book, d["code"]))
            if g is not None:
                hits = sum(name_in_text(n, ts) is not None for n in g.c_name_chn.unique())
                if hits > best[0]:
                    best = (hits, book)
        rows.append(dict(code=d["code"], rid=d["rid"], 墓主="/".join(s["name"] for s in d["subjects"]), book=best[1], cbdb_persons=best[0], matched=best[1] is not None,
                         in_cbdb=any((b, d["code"]) in by for b in ("正编", "续集"))))
    mp = pd.DataFrame(rows); mp.to_csv(os.path.join(HUIBIAN_DIR, "汇编_CBDB映射.csv"), index=False, encoding="utf-8-sig")
    have = set(mp.code)
    miss = src[~src.code.isin(have)].groupby(["book", "code"]).agg(人物=("c_name_chn", lambda x: "、".join(x.unique()))).reset_index()
    miss.to_csv(os.path.join(HUIBIAN_DIR, "CBDB有_我方缺_编号清单.csv"), index=False, encoding="utf-8-sig")
    print("原文 %d 篇: 映射到正编 %d / 续集 %d / 未映射 %d (其中编号不在 CBDB %d, 编号在但人名对不上 %d)"
          % (len(mp), (mp.book == "正编").sum(), (mp.book == "续集").sum(), (~mp.matched).sum(), (~mp.in_cbdb).sum(), (mp.in_cbdb & ~mp.matched).sum()))
    print("CBDB 有而我方缺: 正编 %d 号, 续集 %d 号" % ((miss.book == "正编").sum(), (miss.book == "续集").sum()))


if __name__ == "__main__":
    main()
