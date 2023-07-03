import os
import json
import collections

data_dir = "./pgdata800"

stat = collections.defaultdict(set)

node_stat = collections.defaultdict(set)

def get_plan_dict(path):
    with open(path, 'r') as f:
        jss = f.readlines()
    return [json.loads(s)['Plan'] for s in jss]

def plan_analyze(plan_dict):
    def add_stat(str):
        stat[str].add(plan_dict[str])
    def has_attr(str):
        if str in plan_dict.keys(): return True
        return False
    def try_stat(str):
        if has_attr(str): 
            if type(plan_dict[str]) == type(list()):
                for l in plan_dict[str]:
                    stat[str].add(l)
            else:
                stat[str].add(plan_dict[str])


    add_stat("Node Type")
    try_stat("Parent Relationship")
    try_stat("Join Type")
    try_stat("Sort Method")
    try_stat("Strategy")
    try_stat("Index Name")
    try_stat("Merge Key")
    try_stat("Hash Key")
    try_stat("Sort Key")

    if plan_dict['Node Type'] == 'Gather Motion':
        try:
            a = (plan_dict['Merge Key'])
        except:
            print('M'*10)
    if plan_dict['Node Type'] == 'Redistrubute Motion':
        try:
            a = (plan_dict['Hash Key'])
        except:
            print('H'*10)

    for key in plan_dict.keys():
        node_stat[plan_dict['Node Type']].add(key)

    if has_attr("Plans"):
        for subplan in plan_dict["Plans"]:
            plan_analyze(subplan)



if __name__ == '__main__':
    fnames = [fname for fname in os.listdir(data_dir) if 'txt' in fname]
    for i, fname in enumerate(fnames):
        print(f'query:{i+1}/19 |name:{fname}')
        path = os.path.join(data_dir, fname)
        plan_dicts = get_plan_dict(path)
        for ii, plan in enumerate(plan_dicts):
            # print(f'query:{i+1}/19 |name:{fname} |tree:{ii+1} / 100')
            plan_analyze(plan)

    for k, v in stat.items():
        print(k,":\n",v)
    print('-'*90)
    
    for k, v in node_stat.items():
        print(k,":\n",v)