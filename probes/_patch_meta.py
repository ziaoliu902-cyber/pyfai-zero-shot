import json
import os
for d in ('far_patch_v1', 'hard_patch_v1'):
    base = os.path.join('data', d)
    for sub in sorted(os.listdir(base)):
        p = os.path.join(base, sub, 'meta.json')
        if os.path.isdir(os.path.join(base, sub)) and os.path.exists(p):
            m = json.load(open(p, encoding='utf-8'))
            c = m['config']
            print(d + '/' + sub, 'N=', m.get('total'), 'dist', c.get('dist_range'),
                  'minvis', c.get('min_visible_peaks'), 'hump', c.get('hump_prob'),
                  'counts', c.get('peak_counts_range'), 'bs', c.get('beamstop_prob'))
        elif os.path.exists(p):
            m = json.load(open(p, encoding='utf-8'))
            c = m['config']
            print(d, 'N=', m.get('total'), 'dist', c.get('dist_range'),
                  'minvis', c.get('min_visible_peaks'), 'hump', c.get('hump_prob'),
                  'counts', c.get('peak_counts_range'), 'bs', c.get('beamstop_prob'))
