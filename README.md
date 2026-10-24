# kincalc — Kinship Calculus as Free Supervision

Code for the paper *Kinship Calculus as Free Supervision: Label-Efficient Construction of Genealogical Networks from Historical Biographies* (under review, 2026).

The idea: kinship relations form a small calculus (eight generators F M S D H W B Z, closed under composition, inversion and *re-rooting*). An LLM parser reads one document from the point of view of one person (the ego) and emits `name | path` lines. The calculus then supplies supervision the labels cannot: it **synthesises** exactly-labelled training documents (KinSynth), **verifies** parses by comparing views from different egos (re-rooting agreement), and **composes** relations across documents through shared persons.

The data are in this repository as well: `data/` (aligned corpus, instruction splits, KinSynth corpora, cross-document benchmark, the three verified test references, the 40,905-person tiered network) and `results/` (every system's dev/test outputs and metrics). `DATA.md` documents the formats and provenance; data and results are licensed CC BY-NC-SA 4.0 (`LICENSE-DATA`), code MIT (`LICENSE`).

## Contents

```
kin_algebra.py          the calculus: canonical form, inverse, composition, re-rooting (pure Python, no dependencies)
alignment_pipeline/     epitaph <-> CBDB alignment, step0-step5, with documentation (数据处理说明.md)
parser/                 build_sft_data.py  EpiNet -> instruction data
                        train_lora.py / infer_eval.py  Qwen2.5-7B-Instruct + LoRA parser (train / decode / evaluate)
                        refine_labels.py   KB co-refinement: 2-fold cross-fitting, re-rooting tiers, completed labels
                        make_st.py / pseudo_label.py  consistency-selected self-training
                        A1_consistency_signal.py / conf_baselines.py  confidence signals (re-rooting agreement, log-prob, self-consistency)
                        silver_judge.py    pooled adjudication with an LLM judge (silver reference)
                        metrics_v3.py / metrics_v4_ci.py  item-level metrics under a chosen verified reference, bootstrap CIs
                        N1_build_network.py / N2_marriage.py  network assembly and the marriage analysis
                        docre_tag.py       discriminative token-tagging baseline on the same backbone
                        build_redocred_kin.py  Re-DocRED kinship subset conversion (English)
                        run.sh, gpu_queue.py, gpu_queue_pair.py  job runner and GPU schedulers
kinsynth/               gen_synth_zh.py, gen_synth_en.py  synthetic corpus generators
crossdoc/               build_crossdoc_pairs.py, crossdoc_infer.py, L1_crossdoc_link.py  cross-document benchmark and inference
label_free/             api_pseudo.py, fewshot_api.py  zero-/few-shot LLM pseudo-labelling
evaluation/             export_verify.py  builds the human-verification workbook
```

## Setup

```bash
pip install torch transformers peft openpyxl numpy openai
```

A single 24 GB GPU suffices for every run (Qwen2.5-7B-Instruct in bf16 + LoRA r = 16). Scripts take the base model from `--base` (default `Qwen/Qwen2.5-7B-Instruct`). LLM-API scripts read `LLM_API_KEY` and optionally `LLM_BASE_URL` from the environment.

## Typical runs

```bash
# 1. instruction data from the aligned corpus (TangEpiNet/data/epinet -> sft/G_*.jsonl)
python parser/build_sft_data.py

# 2. baseline parser: train on all labels, evaluate on dev
bash parser/run.sh 0 G G --train sft/G_train.jsonl

# 3. KinSynth pre-training, then fine-tune on 10% of the labels
bash parser/run.sh 0 SYN G --train sft/kinsynth_zh_ego_root_6k.jsonl --epochs 1.5
bash parser/run.sh 0 SYN_10pct G --train sft/G_train.jsonl --init runs/SYN --frac 0.1 --epochs 6

# 4. KB co-refinement -> completed labels -> retrain
python parser/refine_labels.py
bash parser/run.sh 0 Aall G --train sft/Aall_train.jsonl

# 5. item-level metrics on the test set under the expert / AI / consensus reference
T1=t1_expert.tsv OUTSUF=_expert python parser/metrics_v3.py
python parser/metrics_v4_ci.py G Aall STcons SYN_full

# 6. cross-document inference and the network
python crossdoc/crossdoc_infer.py --mode algebra
python parser/N1_build_network.py && python parser/N2_marriage.py
```

Paths inside the scripts follow the original project layout (`sft/`, `runs/`, `runs/verify*`); point them at the folders of the data repository.

## Licence

Code: MIT (`LICENSE`). Data and results: CC BY-NC-SA 4.0 (`LICENSE-DATA`), because the labels derive from the China Biographical Database.

## Citation

```
@inproceedings{kincalc2027,
  title     = {Kinship Calculus as Free Supervision: Label-Efficient Construction of Genealogical Networks from Historical Biographies},
  booktitle = {Proceedings of the ACM Web Conference 2027 (WWW '27)},
  year      = {2027}
}
```
