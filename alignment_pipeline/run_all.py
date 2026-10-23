# -*- coding: utf-8 -*-
"""一键跑完 step1 → step4 (step0 下载和 step5 Tackett 按需单独跑)。用法: python run_all.py"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for s in ["step1_extract_huibian_corpus.py", "step2_map_cbdb.py", "step3_build_epinet.py", "step4_stats_and_checks.py"]:
    print("\n=====", s); subprocess.run([sys.executable, os.path.join(HERE, s)], check=True)
