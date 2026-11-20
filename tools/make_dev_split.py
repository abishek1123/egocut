"""Hold out whole training participants as an honest dev split (validation labels never used for tuning)."""
import json, os, collections
AF = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'actionformer')
DEV = {'P04', 'P22', 'P30'}
for head in ('verb', 'noun'):
    d = json.load(open(f'{AF}/data/epic_kitchens/annotations/epic_kitchens_100_{head}.json'))
    for vid, v in d['database'].items():
        if v['subset'] == 'training' and vid.split('_')[0] in DEV:
            v['subset'] = 'dev'
    json.dump(d, open(f'{AF}/data/epic_kitchens/annotations/egocut_dev_{head}.json', 'w'))
    print(head, collections.Counter(v['subset'] for v in d['database'].values()))
