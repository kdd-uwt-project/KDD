# coding=utf-8

import csv
import numpy as np
import pickle
import json


def get_data():
    train_clicks_file = './data_set_phase1/train_clicks.csv'
    train_queries_file = './data_set_phase1/train_queries.csv'
    train_plans_file = './data_set_phase1/train_plans.csv'

    user_click = []
    clicks_reader = csv.reader(open(train_clicks_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        line[0] = int(line[0])
        line[2] = int(line[2])
        user_click.append(line)

    pickle.dump(user_click, open('./data_set_phase1/train_clicks.pickle', 'wb'))

    user_plans = []
    clicks_reader = csv.reader(open(train_plans_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        line[0] = int(line[0])
        recomends = json.loads(line[2])
        for recomend in recomends:
            cur = []
            cur.append(int(line[0]))
            cur.append(line[1])
            cur.append(float(recomend['distance']))
            cur.append(float(recomend['price']) if not recomend['price'] == '' else 0)
            cur.append(float(recomend['eta']))
            cur.append(int(recomend['transport_mode']))
            user_plans.append(cur)

    pickle.dump(user_plans, open('./data_set_phase1/train_plans.pickle', 'wb'))

    user_queries = []
    clicks_reader = csv.reader(open(train_queries_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        cur = []
        cur.append(int(line[0]))
        cur.append(line[1])
        cur.append(line[2])
        latlong1 = line[3].split(',')
        latlong2 = line[4].split(',')
        cur.append(float(latlong1[0]) if len(latlong1) > 0 else '')
        cur.append(float(latlong1[1]) if len(latlong1) > 1 else '')
        cur.append(float(latlong2[0]) if len(latlong2) > 0 else '')
        cur.append(float(latlong2[1]) if len(latlong2) > 1 else '')
        user_queries.append(cur)

    pickle.dump(user_queries, open('./data_set_phase1/train_queries.pickle', 'wb'))


if __name__ == '__main__':
    get_data()
