import sklearn
import pickle
import bisect
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import pandas as pd
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
        order = 1
        for recomend in recomends:
            cur = []
            cur.append(int(line[0]))
            cur.append(line[1])
            cur.append(float(recomend['distance']))
            cur.append(float(recomend['price']) if not recomend['price'] == '' else '')
            cur.append(float(recomend['eta']))
            cur.append(int(recomend['transport_mode']))
            cur.append(order)
            order += 1
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

def data_gen(hour_m, train_index):
    #load data from files
    print("load data")
    user_click_file = open('./data_set_phase1/train_clicks.pickle',"rb")
    user_click = pickle.load(user_click_file)
    user_click_file.close()

    user_plans_file = open('./data_set_phase1/train_plans.pickle', "rb")
    user_plans = pickle.load(user_plans_file)
    user_plans_file.close()

    #clean dagta
    df_userPlans = pd.DataFrame(user_plans, columns=['Sid','PlanTime','Distance','Price','ETA','TransportMode','Order'])
    #values = [3,5,6]
    #df_userPlans.loc[df_userPlans['TransportMode'].isin(values), 'Price'] = 0
    df_userPlans.loc[df_userPlans['Price'] == '', 'Price'] = 0

    #df_userPlans = df_userPlans.drop(df_userPlans[df_userPlans['Sid'].isin(df_userPlans[df_userPlans['Price']=='']['Sid'])].index)

    #match each plan with the click in session
    df_userClick = pd.DataFrame(user_click, columns = ['Sid','ClickTime', 'TransportMode'])
    df_userPlans_with_click = df_userPlans[['Sid','PlanTime','Distance','Price','ETA','TransportMode','Order']]
    df_userPlans_with_click = pd.merge(df_userPlans_with_click, df_userClick, how='left', left_on='Sid', right_on='Sid')
    df_userPlans_with_click = df_userPlans_with_click.rename(columns = {'TransportMode_x':'TransportMode', 'TransportMode_y':'Click'})

    #mark the plan which does not match with any click information as 0
    #df_userPlans_with_click.loc[df_userPlans_with_click['Click'].isnull(), 'Click'] = 0
    #change type of plantime to datatime
    df_userPlans_with_click['PlanTime'] = pd.to_datetime(df_userPlans_with_click['PlanTime'])
    #set Hour column
    df_userPlans_with_click['Hour'] = getHourMByList(df_userPlans_with_click['PlanTime'].dt.hour, hour_m)

    #set TransportMode as 0 for plans which does not match any click
    df_userPlans_with_click.loc[df_userPlans_with_click['Click'].isnull(), 'Click'] = 0

    #set isSelected
    #set isSelected = 2 when click == 0
    df_userPlans_with_click['isSelected'] = 0
    df_userPlans_with_click.loc[df_userPlans_with_click['TransportMode'] == df_userPlans_with_click['Click'], 'isSelected'] = 1
    df_userPlans_with_click.loc[df_userPlans_with_click['Click']==0, 'isSelected'] = 2
    

    #add new features: the first transportMode in each session
    df_TMFirst = df_userPlans.groupby('Sid').first().reset_index()
    df_TMFirst = df_TMFirst[['Sid', 'TransportMode']].rename(columns={'TransportMode':'FirstTM'})
    df_userPlans_with_click = pd.merge(df_userPlans_with_click, df_TMFirst, how = "left", on=['Sid'])

    print("generate data")
    #create data structure to store all plans to each transportMode
    limit_range =  int(len(df_userPlans_with_click) / 10 * 9)
    data_all_train = df_userPlans_with_click.iloc[:limit_range]
    data_all_test = df_userPlans_with_click.iloc[limit_range:]

    data_transportMode_train = [[] for i in range(11)]
    data_transportMode_label_train = [[] for i in range(11)]

    for index in range(11):
        #for transportMode == 3, dismiss the order
        if index == 2:
            temp_data = data_all_train[data_all_train['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'ETA', 'FirstTM']]
            data_transportMode_label_train[index] = temp_data['isSelected']
        elif index == 3:
            #for transportMode == 4, dismiss the order and price
            temp_data = data_all_train[data_all_train['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]
            data_transportMode_label_train[index] = temp_data['isSelected']
        elif index == 7:
            #for transportMode == 8, dismiss the price
            temp_data = data_all_train[data_all_train['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]
            data_transportMode_label_train[index] = temp_data['isSelected']
        else:
            temp_data = data_all_train[data_all_train['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]
            data_transportMode_label_train[index] = temp_data['isSelected']

    #test data
    data_transportMode_test = data_all_test[['Sid', 'TransportMode', 'Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]
    data_transportMode_label_test = data_all_test[['Sid', 'Click']]
    data_transportMode_label_test = data_transportMode_label_test.drop_duplicates(subset=['Sid'])

    #dump to files
    pickle.dump(data_transportMode_train, open('./data_set_phase1/train_data/data_'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data_transportMode_test, open('./data_set_phase1/test_data/data_'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data_transportMode_label_train, open('./data_set_phase1/train_data/label_'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data_transportMode_label_test, open('./data_set_phase1/test_data/label_'+ str(train_index) + '.pkl', 'wb'))


def train(train_index):
    print("train model");
    models = [GradientBoostingClassifier() for i in range(11)]
    data = pickle.load(open('./data_set_phase1/train_data/data_'+ str(train_index) + '.pkl', 'rb'))
    label = pickle.load(open('./data_set_phase1/train_data/label_'+ str(train_index) + '.pkl', 'rb'))
    for i in range(11):
        print(i)
        models[i].fit(data[i], label[i])

    for i in range(11):
        pickle.dump(models[i], open('./data_set_phase1/model/Xgboost/model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'wb'))

def test(train_index):
    print("get test result")
    models_after_train = []
    data_test_file = open('./data_set_phase1/test_data/data_'+ str(train_index) + '.pkl', 'rb')
    data_test = pickle.load(data_test_file)
    data_test_file.close()
    label_test_file = open('./data_set_phase1/test_data/label_'+ str(train_index) + '.pkl', 'rb')
    label_test = pickle.load(label_test_file)
    label_test_file.close()

    for i in range(11):
        models_after_train.append(pickle.load(open('./data_set_phase1/model/Xgboost/model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'rb')))
    #convert to dataframe
    #test model
    pred = []
    actual = []
    right = 0
    count = 0
    session = data_test.drop_duplicates(subset=['Sid']);
    session = session[['Sid']]
    for sid_index, sid in session.iterrows():
        #for each transportMode
        selected_transportMode = 0
        plans = data_test[data_test['Sid'] == sid['Sid']]
        max_value = 0
        count_not_click = 0
        for plan_index, eplan in plans.iterrows():
            #for each plan
            plan_data = []
            if eplan['TransportMode'] == 3:
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'ETA', 'FirstTM']]]
            elif eplan['TransportMode'] == 4:
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]]
            elif eplan['TransportMode'] == 8:
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]]
            else:
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'Price', 'ETA', 'FirstTM']]]
            result = models_after_train[eplan['TransportMode'] - 1].predict_proba(plan_data)[0]
            cur_max_value = 0
            cur_selected_label = 0
            for i in range(3):
                if result[i] > cur_max_value:
                    cur_max_value = result[i]
                    cur_selected_label = i
            #select the first plan when cur_max_value == max_value
            if cur_selected_label == 1 and cur_max_value > max_value and selected_transportMode != 3 and selected_transportMode != 8:
                selected_transportMode = eplan['TransportMode']
                max_value = cur_max_value
            elif cur_selected_label == 2:
                count_not_click += 1

        result = 0
        if (count_not_click <= (len(plans) / 2)) and selected_transportMode != 3 and selected_transportMode != 8:
            result = selected_transportMode

        pred.append(result)
        actual_click = label_test[label_test['Sid']==sid['Sid']]
        actual.append(actual_click['Click'].values[0])
        if result == actual_click['Click'].values[0]:
            right += 1

        if count % 1000 == 999:
            print(count + 1)
        count += 1

    pickle.dump(pred, open('./data_set_phase1/model/Xgboost/test_pred_result_'+ str(train_index) + '.pkl', 'wb'))
    output = open('./data_set_phase1/output/Xgboost/output_'+ str(train_index) + '.txt', 'w')
    output.write("right:" + str(right) + ", count: " + str(count)) 
    output.write("\n")
    output.write(str(metrics.recall_score(actual, pred, average='micro')))
    output.write("\n")
    output.write(str(metrics.precision_score(actual, pred, average='micro')))
    output.write("\n")
    output.write(str(metrics.f1_score(actual, pred, average='micro')))
    output.write("\n")
    output.write(metrics.classification_report(actual, pred))
    output.close()


def get_result(hour_m, train_index):
    data = pickle.load(open('./data_set_phase1/test_plans.pickle', 'rb'))
    data = pd.DataFrame(data, columns=['Sid','PlanTime','Distance','Price','ETA','TransportMode','Order'])
    data.loc[data['Price']=='', 'Price'] = 0
    data['PlanTime'] = pd.to_datetime(data['PlanTime'])
    #set Hour column
    data['Hour'] = getHourMByList(data['PlanTime'].dt.hour, hour_m)

    queries = pickle.load(open('./data_set_phase1/test_queries.pickle', 'rb'))
    models = []
    for i in range(11):
        models.append(pickle.load(open('./data_set_phase1/model/Xgboost/model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'rb')))

    count = 0
    pred = []
    for esid in queries:
        #for each transportMode
        sid = esid[0]
        selected_transportMode = 0
        plans = data[data['Sid'] == sid]
        max_value = 0
        count_not_click = 0

        for plan_index, eplan in plans.iterrows():
            #for each plan
            plan_data = []
            if eplan['TransportMode'] == 3:
                plan_data = [eplan[['Hour', 'Distance', 'Price', 'ETA']]]
            elif eplan['TransportMode'] == 4:
                plan_data = [eplan[['Hour', 'Distance', 'ETA']]]
            elif eplan['TransportMode'] == 8:
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'ETA']]]
            else:
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'Price', 'ETA']]]
            result = models[eplan['TransportMode'] - 1].predict_proba(plan_data)[0]
            cur_max_value = 0
            cur_selected_label = 2
            for i in range(3):
                if result[i] > cur_max_value:
                    cur_max_value = result[i]
                    cur_selected_label = i
            #select the first plan when cur_max_value == max_value
            if cur_selected_label == 1 and cur_max_value > max_value:
                selected_transportMode = eplan['TransportMode']
                max_value = cur_max_value
            elif cur_selected_label == 2:
                count_not_click += 1

        result = 0
        if (count_not_click < (len(plans) / 2)):
            result = selected_transportMode

        pred.append([sid, result])
        if count % 1000 == 999:
            print(count + 1)
        count += 1

    #output to files
    pred= pd.DataFrame(pred, columns=['Sid', 'TransportMode'])
    pred.to_csv("./data_set_phase1/output/Xgboost/predict_" + str(train_index) + ".csv", index = False, header = True)


def getHourM(hour_mapper):
    hour_m = {}
    for hour_mm in hour_mapper:
        for k in range(hour_mm[1][0], hour_mm[1][1]+ 1):
            hour_m[k] = hour_mm[0]
    return hour_m

def getHourMByList(hourSeries, hour_function):
    hour_m = []
    for eachHour in hourSeries:
        hour = hour_function[eachHour]
        hour_m.append(hour)
    return hour_m

if __name__ == '__main__':

    hour_mapper = [
        [0, [8, 10]],
        [1, [11, 16]],
        [2, [17, 21]],
        [3, [22, 23]],
        [4, [0, 7]]
    ]

    hour_m3 = getHourM(hour_mapper)

    #get_data()
    data_gen(hour_m3, 'WithOrder_CheckWith3and8_withFirstTM')
    train('WithOrder_CheckWith3and8_withFirstTM')
    test('WithOrder_CheckWith3and8_withFirstTM')
    #get_result(hour_m3, 3)


    # get_result()