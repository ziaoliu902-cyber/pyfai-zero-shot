import json
a = json.load(open(r'data/practice_patch_v1/practice/meta.json', encoding='utf-8'))['config']
b = json.load(open(r'data/practice_patch_v1/regular/meta.json', encoding='utf-8'))['config']
c = json.load(open(r'data/train_v2/meta.json', encoding='utf-8'))['config']
d = json.load(open(r'data/finetune_v1/meta.json', encoding='utf-8'))['config']
keys = ('detector_name', 'binning', 'wavelength', 'dist_range',
        'poni_frac_range', 'rot1_range', 'rot2_range', 'rot3_range')
for k in keys:
    ok = (a.get(k) == b.get(k) == c.get(k))
    flag = 'OK  ' if ok else 'DIFF'
    print(f'{flag} {k:16s} P={a.get(k)} R={b.get(k)} V2={c.get(k)} FT={d.get(k)}')
