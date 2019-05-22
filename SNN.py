import sklearn
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.input = nn.Linear(6, 40)
        self.hidden = nn.Linear(40, 40)
        self.output = nn.Linear(40, 2)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = nn.Sigmoid(x)
        return x


def data_gen():
    train_plans = pickle.load(open('./data_set_phase1/train_plans.pickle', 'rb'))
    train_clicks = pickle.load(open('./data_set_phase1/train_clicks.pickle', 'rb'))

    train_plans.sort(key=lambda x: x[0])
    train_clicks.sort(key=lambda x: x[0])

    print(train_plans[0])
    print(train_clicks[0])

    data = [[], []]
    data_label = [[], []]
    index = 0

    count = 0
    click_index = 0
    for plan in train_plans:
        if index != 1:
            hour = int(plan[1].split(' ')[1].split(':')[0])
            if 8 <= hour <= 10:
                hour = 0
            elif 11 <= hour <= 15:
                hour = 1
            elif 16 <= hour <= 20:
                hour = 2
            else:
                hour = 3
            target_mode = 0
            if plan[0] == train_clicks[click_index][0]:
                target_mode = train_clicks[click_index][1]
                click_index += 1
            eindex = 0
            minDis = 1e30
            minPrice = 1e30
            minETA = 1e30
            for eplan in plan[2]:
                if eplan['distance'] < minDis and eplan['distance'] != 0:
                    minDis = eplan['distance']
                if eplan['price'] != '':
                    if eplan['price'] < minPrice:
                        minPrice = eplan['price']
                if eplan['eta'] < minETA and eplan['eta'] != 0:
                    minETA = eplan['eta']
            for eplan in plan[2]:
                data[index].append([eplan['transport_mode'] - 1, eindex, hour, eplan['distance'] / minDis, eplan['price'] / minPrice if eplan['price'] != '' else 0, eplan['eta'] / minETA])
                data_label[index].append(1 if target_mode == eplan['transport_mode'] else 0)
                eindex += 1
                if eindex > 3:
                    break
            count += 1
            if count > len(train_plans) / 5 * 4:
                index = 1
        else:
            target_mode = 0
            if plan[0] == train_clicks[click_index][0]:
                target_mode = train_clicks[click_index][1]
                click_index += 1
            data[index].append(plan)
            data_label[index].append(target_mode)

    pickle.dump(data[0], open('./train_data/nn_data.pkl', 'wb'))
    pickle.dump(data[1], open('./test_data/data.pkl', 'wb'))
    pickle.dump(data_label[0], open('./train_data/nn_label.pkl', 'wb'))
    pickle.dump(data_label[1], open('./test_data/label.pkl', 'wb'))


def train():
    models = [GradientBoostingClassifier() for i in range(11)]
    # models[0] = RandomForestClassifier()
    data = pickle.load(open('./train_data/data.pkl', 'rb'))
    label = pickle.load(open('./train_data/label.pkl', 'rb'))
    for i in range(11):
        print(i)
        models[i].fit(data[i], label[i])

    for i in range(11):
        pickle.dump(models[i], open('./model/Xgboost/model_for_transport_%d.pkl' % i, 'wb'))


def test():
    models = []
    data = pickle.load(open('./test_data/data.pkl', 'rb'))
    label = pickle.load(open('./test_data/label.pkl', 'rb'))
    for i in range(11):
        models.append(pickle.load(open('./model/Xgboost/model_for_transport_%d.pkl' % i, 'rb')))
    pred = []
    right = 0
    count = 0
    for query in data:
        # print(len(query[2]))
        max_value = 0
        max_index = 0
        pindex = 0
        minDis = 1e30
        minPrice = 1e30
        minETA = 1e30
        for eplan in query[2]:
            if eplan['distance'] < minDis and eplan['distance'] != 0:
                minDis = eplan['distance']
            if eplan['price'] != '':
                if eplan['price'] < minPrice:
                    minPrice = eplan['price']
            if eplan['eta'] < minETA and eplan['eta'] != 0:
                minETA = eplan['eta']
        for plan in query[2]:
            hour = int(query[1].split(' ')[1].split(':')[0])
            if 8 <= hour <= 10:
                hour = 0
            elif 11 <= hour <= 15:
                hour = 1
            elif 16 <= hour <= 20:
                hour = 2
            else:
                hour = 3
            temp = [[pindex, hour, plan['distance'] / minDis, plan['price'] / minPrice if plan['price'] != '' else 0,
                     plan['eta'] / minETA]]

            result = models[plan['transport_mode'] - 1].predict_proba(temp)[0]
            if result[1] > max_value:
                max_value = result[1]
                max_index = plan['transport_mode']
            pindex += 1

        result = 0
        if max_value > 0.15:
            result = max_index
        # else:
        #     print('haha')

        pred.append(result)
        if result == label[count]:
            right += 1

        if count % 1000 == 999:
            print(count + 1)
        count += 1

    pickle.dump(pred, open('./model/Xgboost/test_pred_result.pkl', 'wb'))
    print(right, count)
    print(metrics.recall_score(label, pred, average='micro'))
    print(metrics.precision_score(label, pred, average='micro'))
    print(metrics.f1_score(label, pred, average='micro'))
    print(metrics.classification_report(label, pred))


def get_result():
    data = pickle.load(open('./data_set_phase1/test_plans.pickle', 'rb'))
    models = []
    f = open("xgboost.csv", 'w')
    f.write('"sid","recommend_mode"\n')
    for i in range(11):
        models.append(pickle.load(open('./model/Xgboost/model_for_transport_%d.pkl' % i, 'rb')))
    count = 0
    for query in data:
        # print(len(query[2]))
        max_value = 0
        max_index = 0
        pindex = 0
        minDis = 1e30
        minPrice = 1e30
        minETA = 1e30
        for eplan in query[2]:
            if eplan['distance'] < minDis and eplan['distance'] != 0:
                minDis = eplan['distance']
            if eplan['price'] != '':
                if eplan['price'] < minPrice:
                    minPrice = eplan['price']
            if eplan['eta'] < minETA and eplan['eta'] != 0:
                minETA = eplan['eta']
        for plan in query[2]:
            hour = int(query[1].split(' ')[1].split(':')[0])
            if 8 <= hour <= 10:
                hour = 0
            elif 11 <= hour <= 15:
                hour = 1
            elif 16 <= hour <= 20:
                hour = 2
            else:
                hour = 3
            temp = [[pindex, hour, plan['distance'] / minDis, plan['price'] / minPrice if plan['price'] != '' else 0, plan['eta'] / minETA]]

            result = models[plan['transport_mode']-1].predict_proba(temp)[0]
            if result[1] > max_value:
                max_value = result[1]
                max_index = plan['transport_mode']
            pindex += 1
        count += 1
        if count % 1000 == 999:
            print(count)

        if max_value > 0.1:
            f.write('"%d","%d"\n' % (query[0], max_index))
        else:
            f.write('"%d","%d"\n' % (query[0], 0))


if __name__ == '__main__':
    data_gen()
    # train()
    # test()
    # get_result()