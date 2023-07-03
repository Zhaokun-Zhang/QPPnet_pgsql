import os
import json
from collections import Counter

data_dir = './pgdata800'

def hash(plan_dict):
    res = plan_dict['Node Type']
    if 'Plans' in plan_dict:
        for chld in sorted(plan_dict['Plans'], key=lambda p:p['Node Type']):
            res += hash(chld)
    return res

if __name__ == '__main__':
    fpaths = [os.path.join(data_dir, fname) for fname in 
              os.listdir(data_dir) if 'txt' in fname]
    stat = ''
    for i, path in enumerate(fpaths):
        print(i+1, '/22')
        with open(path, 'r') as f:
            lines = f.readlines()
        plandicts = [json.loads(line)['Plan'] for line in lines]
        hash_strs = []
        for plan in plandicts:
            hash_strs.append(hash(plan))
        c = Counter(hash_strs)
        for tuple in c.most_common():
            stat += f"{i+1}: {tuple[0]}\n\t{tuple[1]}\n"
    with open('./hash_grp.txt','w') as f:
        f.write(stat)

            