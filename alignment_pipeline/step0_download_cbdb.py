# -*- coding: utf-8 -*-
"""
Step 0-A  下载 CBDB 官方 SQLite 全库 (Hugging Face: cbdb/cbdb-sqlite)

  latest.json 给出当前版本文件名、sha256、下载地址 → 下载 zip → 校验 → 解压到 data/external/1_史学家人工关系库/CBDB/
  已存在同名 sqlite 时跳过。约 140 MB 下载, 解压 590 MB。
"""
import os, sys, json, hashlib, zipfile, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
from common import CBDB_DIR

LATEST = "https://huggingface.co/datasets/cbdb/cbdb-sqlite/resolve/main/latest.json"
os.makedirs(CBDB_DIR, exist_ok=True)
meta = json.load(urllib.request.urlopen(LATEST))
print("最新版本:", meta)
sqlite_path = os.path.join(CBDB_DIR, meta["sqlite_filename"])
if os.path.exists(sqlite_path):
    print("已存在, 跳过:", sqlite_path); sys.exit()
zip_path = os.path.join(CBDB_DIR, os.path.basename(meta["huggingface_path"]))
print("下载", meta["huggingface_url"]); urllib.request.urlretrieve(meta["huggingface_url"], zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(CBDB_DIR)
h = hashlib.sha256(open(sqlite_path, "rb").read()).hexdigest()
print("sha256 校验:", "OK" if h == meta["sha256"] else "不一致!")
os.remove(zip_path)
print("完成:", sqlite_path, "\n注意: common.CBDB_SQLITE 的文件名要与之一致")
