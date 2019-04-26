import json, numpy

data_dir = "data_set_phase1"


def loader(mode="train"):
    train = {}
    if mode == "test":
        f = open("%s/test_plans.csv" % (data_dir), 'r')
    else:
        f = open("%s/train_plans.csv" % (data_dir), 'r')
    lines = f.readlines()
    for line in lines:
        tp = line.strip().split(",", 2)
        if tp[0] == '"sid"':
            continue
        sid = tp[0].replace('"', '')
        plantime = tp[1].replace('"', '')
        _plans = tp[2]
        _plans = _plans.replace('""', '"')
        _plans = _plans[1:-1]
        plans = json.loads(_plans)
        for plan in plans:
            if plan["price"] == "":
                plan["price"] = 0
        train[sid] = {"plan_time": plantime, "plans": plans}
    f.close()
    if mode == "test":
        f = open("%s/test_queries.csv" % (data_dir), 'r')
    else:
        f = open("%s/train_queries.csv" % (data_dir), 'r')
    lines = f.readlines()
    for line in lines:
        tp = line.strip().split(",")
        if tp[0] == '"sid"':
            continue
        sid = tp[0].replace('"', '')
        pid = tp[1].replace('"', '')
        if pid == "":
            pid = "-1"
        reqtime = tp[2].replace('"', '')
        o1 = tp[3].replace('"', '')
        o2 = tp[4].replace('"', '')
        d1 = tp[5].replace('"', '')
        d2 = tp[6].replace('"', '')
        if sid in train:
            entry = train[sid]
            entry["pid"] = pid
            entry["req_time"] = reqtime
            entry["o"] = (o1, o2)
            entry["d"] = (d1, d2)
    f.close()
    return train


test = loader("test")
f = open("baseline.csv", 'w')
f.write('"sid","recommend_mode"\n')
for sid in test:
    plans = test[sid]["plans"]
    minpr = 999999
    mintm = 999999
    total_bal = 999999
    minmode = -1
    for plan in plans:
        ttb = plan["eta"] + 2 * plan['price']
        if ttb < total_bal:
            total_bal = ttb
            minmode = plan["transport_mode"]
    f.write('"%s","%d"\n' % (sid, minmode))
f.close()
