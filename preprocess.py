# coding=utf-8

import csv
import numpy as np
import pickle
import json


def get_data():
    train_clicks_file = './data_set_phase1/train_clicks.csv'
    train_queries_file = './data_set_phase1/train_queries.csv'
    train_plans_file = './data_set_phase1/train_plans.csv'
    test_plans_file = './data_set_phase1/test_plans.csv'

    user_click = []
    clicks_reader = csv.reader(open(train_clicks_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        line[0] = int(line[0])
        line[1] = int(line[2])
        user_click.append(line)

    pickle.dump(user_click, open('./data_set_phase1/train_clicks.pickle', 'wb'))

    user_plans = []
    clicks_reader = csv.reader(open(train_plans_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        line[0] = int(line[0])
        line[2] = json.loads(line[2])
        user_plans.append(line)

    pickle.dump(user_plans, open('./data_set_phase1/train_plans.pickle', 'wb'))

    user_plans = []
    clicks_reader = csv.reader(open(test_plans_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        line[0] = int(line[0])
        line[2] = json.loads(line[2])
        user_plans.append(line)

    pickle.dump(user_plans, open('./data_set_phase1/test_plans.pickle', 'wb'))

    user_queries = []
    clicks_reader = csv.reader(open(train_queries_file, 'r'))
    for line in clicks_reader:
        if clicks_reader.line_num == 1:
            continue
        line[0] = int(line[0])
        line[3] = list(map(float, line[3].split(',')))
        line[4] = list(map(float, line[4].split(',')))
        user_queries.append(line)

    pickle.dump(user_queries, open('./data_set_phase1/train_queries.pickle', 'wb'))


if __name__ == '__main__':
    get_data()
