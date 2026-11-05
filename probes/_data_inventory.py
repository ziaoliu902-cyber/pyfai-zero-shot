# -*- coding: utf-8 -*-
"""一次性数据盘点：data/ 下所有 meta.json 关键字段对齐表（只读）"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# (显示名, meta 相对路径)
TARGETS = [
    ("train_v1",                "train_v1/meta.json"),
    ("train_v2",                "train_v2/meta.json"),
    ("train_v3",                "train_v3/meta.json"),
    ("far_patch_v1",            "far_patch_v1/meta.json"),
    ("hard_patch_v1/near_mix",  "hard_patch_v1/near_mix/meta.json"),
    ("hard_patch_v1/agbh_far",  "hard_patch_v1/agbh_far/meta.json"),
    ("practice/practice",       "practice_patch_v1/practice/meta.json"),
    ("practice/regular",        "practice_patch_v1/regular/meta.json"),
    ("finetune_v1",             "finetune_v1/meta.json"),
]

KEYS = ["dist_range", "forced_phase", "min_visible_peaks", "hump_prob",
        "peak_counts_range", "shard_size"]

def fmt(v):
    if v is None:
        return "-"
    return str(v)

for name, rel in TARGETS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"== {name}: META MISSING ({rel})")
        continue
    m = json.load(open(p, encoding="utf-8"))
    cfg = m.get("config", {})
    total = m.get("total", m.get("actual", "?"))
    counts = m.get("counts", {})
    nz = {k: v for k, v in counts.items() if v}  # 只显示非零相
    print(f"== {name}  (n={total}, shards={len([f for f in os.listdir(os.path.dirname(p)) if f.endswith('.npz')])})")
    print(f"   created : {m.get('created','?')}   seed: {m.get('seed', m.get('split_seed','?'))}")
    print(f"   phases  : {nz}")
    for k in KEYS:
        print(f"   {k:20s}: {fmt(cfg.get(k))}")
    prov = m.get("provenance")
    if prov:
        for k, v in prov.items():
            if isinstance(v, dict):
                print(f"   prov/{k}: n={v.get('n')} note={v.get('note','')}")
            else:
                print(f"   prov/{k}: {v}")
    print()
