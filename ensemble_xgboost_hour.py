import sklearn
import pickle
import bisect
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics



def data_gen(hour_m, train_index):
    train_plans = pickle.load(open('./data_set_phase1/train_plans.pickle', 'rb'))
    train_clicks = pickle.load(open('./data_set_phase1/train_clicks.pickle', 'rb'))

    train_plans.sort(key=lambda x: x[0])
    train_clicks.sort(key=lambda x: x[0])

    print(train_plans[0])
    print(train_clicks[0])

    data = [[[] for i in range(11)], []]
    data_label = [[[] for i in range(11)], []]
    index = 0

    count = 0
    click_index = 0
    for plan in train_plans:
        #generate data for traning model
        if index != 1:
            hour = int(plan[1].split(' ')[1].split(':')[0])
            hour = hour_m[hour]
            target_mode = 0
            if plan[0] == train_clicks[click_index][0]:
                target_mode = train_clicks[click_index][1]
                click_index += 1
            eindex = 0
            minDis = 1e30
            minPrice = 1e30
            minETA = 1e30
            for eplan in plan[2]:
                # calculate relative value based on minimum value on each feature
                if eplan['distance'] < minDis and eplan['distance'] != 0:
                    minDis = eplan['distance']
                if eplan['price'] != '':
                    if eplan['price'] < minPrice:
                        minPrice = eplan['price']
                if eplan['eta'] < minETA and eplan['eta'] != 0:
                    minETA = eplan['eta']
            for eplan in plan[2]:
                #index == 0
                data[index][eplan['transport_mode']-1].append([eindex, hour, eplan['distance'] / minDis, eplan['price'] / minPrice if eplan['price'] != '' else 0, eplan['eta'] / minETA])
                data_label[index][eplan['transport_mode']-1].append(1 if target_mode == eplan['transport_mode'] else 0)
                eindex += 1
                if eindex > 3:
                    break
            count += 1
            if count > len(train_plans) / 5 * 4:
                index = 1
        else:
            #generate data for test
            target_mode = 0
            if plan[0] == train_clicks[click_index][0]:
                target_mode = train_clicks[click_index][1]
                click_index += 1
            data[index].append(plan)
            data_label[index].append(target_mode)

    pickle.dump(data[0], open('./train_data/data'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data[1], open('./test_data/data'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data_label[0], open('./train_data/label'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data_label[1], open('./test_data/label'+ str(train_index) + '.pkl', 'wb'))


def train(train_index):
    models = [GradientBoostingClassifier() for i in range(11)]
    # models[0] = RandomForestClassifier()
    data = pickle.load(open('./train_data/data'+ str(train_index) + '.pkl', 'rb'))
    label = pickle.load(open('./train_data/label'+ str(train_index) + '.pkl', 'rb'))
    for i in range(11):
        print(i)
        models[i].fit(data[i], label[i])

    for i in range(11):
        pickle.dump(models[i], open('./model/Xgboost/model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'wb'))


def test(hour_m, train_index):
    models = []
    data = pickle.load(open('./test_data/data'+ str(train_index) + '.pkl', 'rb'))
    label = pickle.load(open('./test_data/label'+ str(train_index) + '.pkl', 'rb'))
    for i in range(11):
        models.append(pickle.load(open('./model/Xgboost/model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'rb')))
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
            hour = hour_m[hour]
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

    pickle.dump(pred, open('./model/Xgboost/test_pred_result'+ str(train_index) + '.pkl', 'wb'))
    output = open('./output/Xgboost/output'+ str(train_index) + '.txt', 'wb')
    output.write("right:" + str(right) + ", count: " + str(count)) 
    output.write("\n")
    output.write(str(metrics.recall_score(label, pred, average='micro')))
    output.write("\n")
    output.write(str(metrics.precision_score(label, pred, average='micro')))
    output.write("\n")
    output.write(str(metrics.f1_score(label, pred, average='micro')))
    output.write("\n")
    output.write(metrics.classification_report(label, pred))
    output.close()


def get_result(hour_m, train_index):
    data = pickle.load(open('./data_set_phase1/test_plans.pickle', 'rb'))
    models = []
    f = open("xgboost.csv", 'w')
    f.write('"sid","recommend_mode"\n')
    for i in range(11):
        models.append(pickle.load(open('./model/Xgboost/model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'rb')))
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
            hour = hour_m[hour]
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

def getHourM(hour_mapper):
    hour_m = {}
    for hour_mm in hour_mapper:
        for k in range(hour_mm[1][0], hour_mm[1][1]+ 1):
            hour_m[k] = hour_mm[0]
    return hour_m

if __name__ == '__main__':
    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 15]],
        [2, [16, 20]],
        [3, [21, 23]],
        [3, [0, 7]]
    ]

    hour_m1 = getHourM(hour_mapper)

    #data_gen(hour_m1, 1)
    #train(1)
    #test(hour_m1, 1)

    hour_mapper = [
        [0, [6, 10]],
        [1, [11, 16]],
        [2, [17, 21]],
        [3, [22, 23]],
        [3, [0, 5]]
    ]

    hour_m2 = getHourM(hour_mapper)

    #data_gen(hour_m2, 2)
    #train(2)
    #test(hour_m2, 2)

    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 16]],
        [2, [17, 21]],
        [3, [22, 23]],
        [4, [0, 7]]
    ]

    hour_m3 = getHourM(hour_mapper)

    #data_gen(hour_m3, 3)
    #train(3)
    #test(hour_m3, 3)
    #get_result(hour_m3, 3)

    hour_mapper = [
        [0, [7, 9]],
        [1, [10, 11]],
        [2, [12, 16]],
        [3, [17, 19]],
        [4, [19, 22]],
        [5, [23, 24]],
        [5, [0, 6]],
    ]

    hour_m4 = getHourM(hour_mapper)

    #data_gen(hour_m4, 4)
    #train(4)
    #test(hour_m4, 4)

    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 16]],
        [2, [17, 22]],
        [3, [23, 24]],
        [4, [0, 7]]
    ]

    hour_m5 = getHourM(hour_mapper)

    #data_gen(hour_m5, 5)
    #train(5)
    #test(hour_m5, 5)

    hour_mapper = [
        [0, [7, 10]],
        [1, [11, 16]],
        [2, [17, 21]],
        [3, [22, 23]],
        [4, [0, 6]]
    ]

    hour_m6 = getHourM(hour_mapper)

    #data_gen(hour_m6, 6)
    #train(6)
    #test(hour_m6, 6)

    hour_mapper = [
        [0, [0, 0]],
        [1, [1, 1]],
        [2, [2, 2]],
        [3, [3, 3]],
        [4, [4, 4]],
        [5, [5, 5]],
        [6, [6, 6]],
        [7, [7, 7]],
        [8, [8, 8]],
        [9, [9, 9]],
        [10, [10, 10]],
        [11, [11, 11]],
        [12, [12, 12]],
        [13, [13, 13]],
        [14, [14, 14]],
        [15, [15, 15]],
        [16, [16, 16]],
        [17, [17, 17]],
        [18, [18, 18]],
        [19, [19, 19]],
        [20, [20, 20]],
        [21, [21, 21]],
        [22, [22, 22]],
        [23, [23, 23]]
    ]

    hour_m7 = getHourM(hour_mapper)

    #data_gen(hour_m7, 7)
    #train(7)
    #test(hour_m7, 7)

    hour_mapper = [
        [0, [8, 11]],
        [1, [12, 16]],
        [2, [17, 21]],
        [3, [22, 23]],
        [4, [0, 7]]
    ]

    hour_m8 = getHourM(hour_mapper)

    #data_gen(hour_m8, 8)
    #train(8)
    #test(hour_m8, 8)

    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 15]],
        [2, [16, 21]],
        [3, [22, 23]],
        [4, [0, 7]]
    ]

    hour_m9 = getHourM(hour_mapper)

    #data_gen(hour_m9, 9)
    #train(9)
    #test(hour_m9, 9)

    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 16]],
        [2, [17, 20]],
        [3, [21, 23]],
        [4, [0, 7]]
    ]

    hour_m10 = getHourM(hour_mapper)

    #data_gen(hour_m10, 10)
    #train(10)
    #test(hour_m10, 10)

    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 16]],
        [2, [17, 21]],
        [3, [22, 23]],
        [4, [6, 7]],
        [5, [0, 5]]
    ]

    hour_m11 = getHourM(hour_mapper)

    data_gen(hour_m11, 11)
    train(11)
    test(hour_m11, 11)

    # get_result()