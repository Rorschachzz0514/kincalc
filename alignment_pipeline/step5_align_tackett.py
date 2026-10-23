# -*- coding: utf-8 -*-
"""
Step 5 (可选)  把 Tackett TBDB 1.5 的关系/姻亲/撰者对齐到我方原文 —— 补 CBDB 没有的姻亲郡望、幕僚、撰书者

Tackett 库没有汇编编号, 只能按 姓名 + 年份 对: People(姓+名) 与我方 姓名(转繁) 相同, 且 Epitaphs.Burial date 与我方 去世年 相差 ≤3 年。
输入  TBDB_DIR/tbdb015_csv/{People,Relations,Epitaphs,Affines,Authors}.csv, RAW_XLSX
输出  TBDB_DIR/对齐数据集/tbdb_text2graph.json (每篇: 原文 + 墓主属性 + 墓志属性 + relations + affines + authors)
      TBDB_DIR/对齐数据集/tbdb_relations_flat.csv
关系方向: Relations 里 "Subject 的 Relationship 是 Target"; 类型 S 子 D 女 W 妻 Sil 女婿 Cl 门客/幕僚; Relationship type Ag 宗亲 / Af 姻亲 / Ot 其他
"""
import os, re, sys, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from common import TBDB_DIR, RAW_XLSX, t2t


def main():
    D = os.path.join(TBDB_DIR, "tbdb015_csv"); rd = lambda t: pd.read_csv(os.path.join(D, t + ".csv"), encoding="utf-8-sig", low_memory=False)
    P, R, E, A, Au = rd("People"), rd("Relations"), rd("Epitaphs"), rd("Affines"), rd("Authors")
    P["name"] = (P["Surname"].fillna("").astype(str) + P["Given name"].fillna("").astype(str)).str.replace(r"[\s·]", "", regex=True)
    nm = P.set_index("ID")["name"].to_dict(); pinfo = P.set_index("ID")
    EP = E.merge(P[["ID", "name"]], left_on="PersonID", right_on="ID", how="left", suffixes=("", "_p"))
    EP["bd"] = pd.to_numeric(EP["Burial date"], errors="coerce"); EP["n2"] = EP["name"].map(t2t)
    e = pd.read_excel(RAW_XLSX); e = e.loc[:, ~e.columns.duplicated()]; e["rid"] = range(len(e))
    e["n"] = e["姓名"].map(lambda x: "" if pd.isna(x) else t2t(re.sub(r"[\s·]", "", str(x)))); e["dy"] = pd.to_numeric(e["去世年"], errors="coerce")
    m = e[e["n"].str.len() >= 2].merge(EP, left_on="n", right_on="n2"); m = m[(m["bd"] - m["dy"]).abs() <= 3].drop_duplicates("rid")
    recs, flat = [], []
    for r in m.itertuples():
        pid, eid = int(r.PersonID), int(r.ID_y) if "ID_y" in m.columns else int(r.ID)
        rel = R[(R["SubjectID"] == pid) | (R["TargetID"] == pid)]
        rels = [dict(head=int(x.SubjectID), head_name=nm.get(x.SubjectID), rel=x.Relationship, rel_type=x._5, tail=int(x.TargetID), tail_name=nm.get(x.TargetID), source=x.References) for x in rel.itertuples()]
        aff = [dict(surname=x["Surname"], given=None if pd.isna(x["Given name"]) else x["Given name"], choronym=x["Choronym"], office=x["Office"], relation_to_subject=x["Relationship to deceased"]) for _, x in A[A["linkMZM"] == eid].iterrows()]
        aut = [dict(job=x["Job"], surname=x["Surname"], given=None if pd.isna(x["Given name"]) else x["Given name"], choronym=x["Choronym"], office=x["Office"], relation_to_subject=x["Relationship to deceased"]) for _, x in Au[Au["linkMZM"] == eid].iterrows()]
        pi = pinfo.loc[pid]; ep = E[E["ID"] == eid].iloc[0]
        recs.append(dict(rid=int(r.rid), 编号=str(r.编号), 墓主=str(r.姓名), 原文=str(r.原文),
                         tbdb_person=dict(id=pid, name=pi["name"], office=pi["Office"], elite_type=pi["Elite Type"], death_year=pi["Death Year"], clan=pi["Clan ID"]),
                         epitaph=dict(burial=ep["Burial date"], choronym=ep["Choronym"], burial_prefecture=ep["Burial prefecture"], sons=ep["Number of sons"], daughters=ep["Number of daughters"]),
                         relations=rels, affines=aff, authors=aut))
        for x in rels: flat.append(dict(rid=int(r.rid), 墓主=str(r.姓名), head=x["head_name"], 关系=x["rel"], 类型=x["rel_type"], tail=x["tail_name"], 出处=x["source"]))
        for x in aff: flat.append(dict(rid=int(r.rid), 墓主=str(r.姓名), head=str(x["surname"]) + (x["given"] or "氏"), 关系="AFFINE:" + str(x["relation_to_subject"]), 类型="Af", tail=str(r.姓名), 出处="Affines"))
        for x in aut: flat.append(dict(rid=int(r.rid), 墓主=str(r.姓名), head=str(x["surname"]) + (x["given"] or ""), 关系="AUTHOR:%s/%s" % (x["job"], x["relation_to_subject"]), 类型="Ot", tail=str(r.姓名), 出处="Authors"))
    out = os.path.join(TBDB_DIR, "对齐数据集"); os.makedirs(out, exist_ok=True)
    json.dump(recs, open(os.path.join(out, "tbdb_text2graph.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    pd.DataFrame(flat).to_csv(os.path.join(out, "tbdb_relations_flat.csv"), index=False, encoding="utf-8-sig")
    print("Tackett 对齐: %d 篇, 关系+姻亲+撰者 %d 条 (每篇 %.1f)" % (len(recs), len(flat), len(flat) / max(len(recs), 1)))


if __name__ == "__main__":
    main()
