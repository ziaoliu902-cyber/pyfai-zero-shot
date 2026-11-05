# -*- coding: utf-8 -*-
"""建 v4c/v4d 微调数据（修正 merge_ft.py 的 config 抄写 bug，2026-10-27）。

产物三个目录：
  data/far_replay_v1  远距重放段 6000 行（从 far_patch_v1 剔泄漏后抽行、重写 npz 分片）
  data/finetune_v2c   20000 行 = practice 12k + regular 8k   ← v4c（纯修编码，单变量）
  data/finetune_v2    26000 行 = v2c + far_replay 6k          ← v4d（+远距重放）

修 bug（对照 docs/对话记录_2026-10-26.md）：
  1) config 不再抄任何单一源：除 dist_range（取并集=[0.1,3.0]，与 train_v3 一致，
     base_v3 权重可直接复用）外，全部字段逐源 assert 相同；外观三字段
     (min_visible_peaks/hump_prob/peak_counts_range) 取 regular 标准值，
     practice 段的实际值记入 provenance.config_overrides。
  2) 远距重放剔泄漏：far_patch_v1 行 j == train_v2 全局行 50000+j == train_v3
     全局行 50000+j（audit_train_v3 已验证逐位恒等）。剔除落在
     (train_v2 val∪test) ∪ (train_v3 val∪test) 的行 → 微调集不污染任何评测卷。
  3) 标签越界自检：打包后用并集常数 param_to_norm，assert |y|<=1+1e-3 全部成立
     （v1 的 399 行越界事故防复发）。

用法：python merge_ft2.py [--force]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.join(ROOT, "data", "practice_patch_v1", "practice")
REGULAR = os.path.join(ROOT, "data", "practice_patch_v1", "regular")
FARSRC = os.path.join(ROOT, "data", "far_patch_v1")
FAR_REPLAY = os.path.join(ROOT, "data", "far_replay_v1")
FT2C = os.path.join(ROOT, "data", "finetune_v2c")
FT2 = os.path.join(ROOT, "data", "finetune_v2")

FAR_WANT = 6000            # 远距重放行数
FAR_OFF_IN_V2 = 50000      # far_patch_v1 行 j 在 train_v2/v3 里的全局行号偏移
SELECT_SEED = 20261027     # 远距重放抽行种子（固定，可复现）
APPEAR_KEYS = ("min_visible_peaks", "hump_prob", "peak_counts_range")
CHUNK = 2048


def lmeta(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def hardlink(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def shard_list(root):
    return sorted(f for f in os.listdir(root)
                  if f.startswith("shard_") and f.endswith(".npz"))


def shard_lens(root, names):
    out = []
    for fn in names:
        with np.load(os.path.join(root, fn)) as z:
            out.append(int(len(z["params"])))
    return out


def fresh_dir(dst, force):
    if os.path.isdir(dst):
        if not force:
            raise SystemExit(f"{dst} 已存在；确认重来请加 --force")
        for f in os.listdir(dst):
            os.remove(os.path.join(dst, f))
    os.makedirs(dst, exist_ok=True)


def union_config(src_metas, appear_ref):
    """程序化构造并集 config：非 dist_range 字段全源必须相同，dist_range 取并集。"""
    cfgs = [m["config"] for m in src_metas]
    keys = set(cfgs[0])
    for c in cfgs[1:]:
        assert set(c) == keys, "config 字段集合不一致"
    out = {}
    for k in sorted(keys):
        vals = [c[k] for c in cfgs]
        if k == "dist_range":
            out[k] = [min(v[0] for v in vals), max(v[1] for v in vals)]
        elif all(v == vals[0] for v in vals):
            out[k] = vals[0]
        elif k in APPEAR_KEYS:
            out[k] = appear_ref["config"][k]   # 外观字段取 regular 标准值
        else:
            raise AssertionError(f"字段 {k} 各源不一致且无处理规则: {vals}")
    return out


def materialize_far_replay(leak_mask, force):
    """按选中的 far 行重写 npz 分片 → data/far_replay_v1（行序保持 far_patch 原序）。"""
    t0 = time.time()
    all_rows = np.arange(len(leak_mask))
    remain = all_rows[~leak_mask]
    rng = np.random.default_rng(SELECT_SEED)
    sel = np.sort(rng.choice(remain, size=min(FAR_WANT, len(remain)), replace=False))
    print(f"[far] far_patch 15000 行，剔泄漏 {int(leak_mask.sum())} 行，"
          f"抽 {len(sel)} 行 (seed={SELECT_SEED})")

    mf = lmeta(os.path.join(FARSRC, "meta.json"))
    fresh_dir(FAR_REPLAY, force)
    names = shard_list(FARSRC)
    lens = shard_lens(FARSRC, names)
    offs = np.concatenate([[0], np.cumsum(lens)])

    k, done, shard_idx = 0, 0, 0
    for fn, n in zip(names, lens):
        s0, s1 = offs[k], offs[k + 1]
        take = sel[(sel >= s0) & (sel < s1)] - s0
        if len(take):
            with np.load(os.path.join(FARSRC, fn)) as z:
                np.savez(os.path.join(FAR_REPLAY, f"shard_{shard_idx:05d}.npz"),
                         imgs=z["imgs"][take], params=z["params"][take], aux=z["aux"][take])
            shard_idx += 1
            done += len(take)
        k += 1
    assert done == len(sel)
    meta = {
        "config": mf["config"], "total": int(done), "shard_size": 256,
        "counts": {"AgBh": int(done)}, "phase_names": mf["phase_names"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "done",
        "split_seed": 0, "split_ratios": [0.88, 0.06, 0.06],
        "provenance": {
            "far_patch_v1": {"root": "data/far_patch_v1", "n": 15000,
                             "note": f"剔除落在 train_v2 val∪test 与 train_v3 val∪test 的行 "
                                     f"（far 行 j == v2/v3 全局行 {FAR_OFF_IN_V2}+j，剔 {int(leak_mask.sum())} 行），"
                                     f"再以 seed={SELECT_SEED} 无放回抽 {int(done)} 行，行序保持原序"},
            "purpose": "远距重放段：进 finetune_v2（v4d），对 v2/v3 评测卷零泄漏",
        },
    }
    with open(os.path.join(FAR_REPLAY, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[far] far_replay_v1 物化完成 {done} 行  {time.time()-t0:.0f}s")
    return sel


def build(dst, parts, cfg, phase_names, purpose, force):
    """parts: [(src_root, note, config_overrides_dict_or_None)]，顺序=行序。"""
    t0 = time.time()
    print(f"\n=== build {os.path.basename(dst)} ===")
    packs = []
    for root, note, _ in parts:
        names = shard_list(root)
        lens = shard_lens(root, names)
        packs.append((root, names, lens, note))
        print(f"  源 {os.path.relpath(root, ROOT)}: {len(names)} 片 {sum(lens)} 行")
    N = sum(sum(p[2]) for p in packs)
    with np.load(os.path.join(packs[0][0], packs[0][1][0])) as z:
        H, W = z["imgs"].shape[1:]

    fresh_dir(dst, force)
    k = 0
    for root, names, lens, _ in packs:
        for fn in names:
            hardlink(os.path.join(root, fn), os.path.join(dst, f"shard_{k:05d}.npz"))
            k += 1
    print(f"[1/4] 硬链接 {k} 片")

    imgs = np.lib.format.open_memmap(os.path.join(dst, "imgs.npy"), mode="w+",
                                     dtype=np.uint16, shape=(N, H, W))
    params = np.empty((N, 6), np.float32)
    aux = np.empty((N, 5), np.float32)
    i = 0
    for root, names, lens, _ in packs:
        for fn, n in zip(names, lens):
            with np.load(os.path.join(root, fn)) as z:
                imgs[i:i + n] = z["imgs"]
                params[i:i + n] = z["params"]
                aux[i:i + n] = z["aux"]
            i += n
        imgs.flush()
    assert i == N
    np.save(os.path.join(dst, "params.npy"), params)
    np.save(os.path.join(dst, "aux.npy"), aux)
    print(f"[2/4] 打包 {N}x{H}x{W} ({N*H*W*2/1e9:.1f}GB)  {time.time()-t0:.0f}s")

    # ---- 标签越界自检（并集常数） ----
    from torchfai.dataset import param_to_norm, _norm_consts
    y = param_to_norm(params.astype(np.float64), _norm_consts(cfg))
    bad = int((np.abs(y) > 1.0 + 1e-3).sum())
    assert bad == 0, f"标签越界 {bad} 个元素！（v1 事故防复发断言）"
    print(f"[3/4] 标签越界自检通过（|y|max={np.abs(y).max():.4f}）")

    # ---- meta ----
    prov, counts, i = {}, {}, 0
    rejects = 0
    for root, note, overrides in parts:
        m = lmeta(os.path.join(root, "meta.json"))
        names = shard_list(root)
        n = sum(shard_lens(root, names))
        prov[os.path.relpath(root, ROOT).replace("\\", "/")] = {
            "root": os.path.relpath(root, ROOT).replace("\\", "/"),
            "n": n, "seed": m.get("seed"), "rows": f"{i}..{i+n-1}",
            "config_overrides": overrides, "note": note}
        for kk, vv in m.get("counts", {}).items():
            counts[kk] = counts.get(kk, 0) + vv
        rejects += m.get("rejects", 0)
        i += n
    meta = {"config": cfg, "total": N, "shard_size": 256, "counts": counts,
            "phase_names": phase_names, "rejects": rejects,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "done",
            "split_seed": 0, "split_ratios": [0.88, 0.06, 0.06],
            "provenance": dict(prov, purpose=purpose)}
    with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[4/4] meta.json 完成")

    # ---- 切分自检 ----
    from torchfai.dataset import TorchFAIDataset
    sels = {sp: TorchFAIDataset(dst, split=sp)._sel for sp in ("train", "val", "test")}
    for sp, sel in sels.items():
        print(f"  split={sp:5s} N={len(sel)}")
    assert np.intersect1d(sels["train"], sels["val"]).size == 0
    assert np.intersect1d(sels["train"], sels["test"]).size == 0
    assert np.intersect1d(sels["val"], sels["test"]).size == 0
    print(f"自检通过：三切分互斥  总耗时 {time.time()-t0:.0f}s")
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    # ---- 0) 泄漏集：far 行是否落在 v2/v3 的 val∪test ----
    from torchfai.dataset import TorchFAIDataset
    hold = []
    for src in ("train_v2", "train_v3"):
        for sp in ("val", "test"):
            hold.append(TorchFAIDataset(os.path.join(ROOT, "data", src), split=sp)._sel)
    hold_all = np.unique(np.concatenate(hold))
    far_global = FAR_OFF_IN_V2 + np.arange(15000)
    leak_mask = np.isin(far_global, hold_all)
    print(f"泄漏行（far∩[v2 val∪test ∪ v3 val∪test]）: {int(leak_mask.sum())}/15000")

    # ---- 1) 并集 config ----
    mp, mr, mfar = (lmeta(os.path.join(p, "meta.json")) for p in (PRACTICE, REGULAR, FARSRC))
    cfg = union_config([mp, mr, mfar], appear_ref=mr)
    assert cfg["dist_range"] == [0.1, 3.0], f"dist 并集异常: {cfg['dist_range']}"
    print(f"并集 config 构造通过（dist_range={cfg['dist_range']}，外观字段取 regular）")

    prac_over = {k: mp["config"][k] for k in APPEAR_KEYS}

    # ---- 2) 远距重放物化 ----
    sel = materialize_far_replay(leak_mask, a.force)

    # ---- 3) 20k 纯修编码版（v4c） ----
    build(FT2C,
          [(PRACTICE, "实际工况段（微调主体）", prac_over),
           (REGULAR, "常规防遗忘段", None)],
          cfg, mp["phase_names"],
          "v4c 微调集（纯修编码单变量：config=并集[0.1,3.0]，base_v3 可直接初始化；"
          "对照 v4b=finetune_v1 的错位编码）", a.force)

    # ---- 4) 26k +远距重放版（v4d） ----
    build(FT2,
          [(PRACTICE, "实际工况段（微调主体）", prac_over),
           (REGULAR, "常规防遗忘段", None),
           (FAR_REPLAY, "远距重放段（剔泄漏，零污染 v2/v3 评测卷）", None)],
          cfg, mp["phase_names"],
          "v4d 微调集（v4c + 远距重放 6k；远距档占比 6.2%→~35% 预期）", a.force)

    # ---- 5) 终局泄漏自检：finetune_v2 的 far 段(行 20000..25999) 对评测卷 ----
    far_rows_in_ft2 = np.arange(20000, 20000 + len(sel))
    far_v3_global = FAR_OFF_IN_V2 + sel          # 这些行在 train_v3 里的全局行号
    v3_hold = np.unique(np.concatenate([
        TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split=sp)._sel
        for sp in ("val", "test")]))
    n_leak = int(np.isin(far_v3_global, v3_hold).sum())
    assert n_leak == 0, f"far 段仍有 {n_leak} 行落在 train_v3 val/test！"
    v3_train = TorchFAIDataset(os.path.join(ROOT, "data", "train_v3"), split="train")._sel
    n_replay = int(np.isin(far_v3_global, v3_train).sum())
    print(f"\n终局自检：finetune_v2 far 段 {len(sel)} 行 "
          f"∩ v3 val/test = 0 ✅；其中 {n_replay} 行 ({n_replay/len(sel)*100:.0f}%) "
          f"曾在 base_v3 训练集（重放语义，非泄漏）")

    print(f"\n全部完成 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
