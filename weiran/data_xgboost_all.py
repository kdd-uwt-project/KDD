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
    
    print("generate data")
    #create data structure to store all plans to each transportMode
    data_transportMode_train = [[] for i in range(11)]
    data_transportMode_label_train = [[] for i in range(11)]

    for index in range(11):
        #for transportMode == 3, dismiss the order
        if index == 2:
            temp_data = df_userPlans_with_click[df_userPlans_with_click['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'ETA']]
            data_transportMode_label_train[index] = temp_data['isSelected']
        elif index == 3:
            #for transportMode == 4, dismiss the order and price
            temp_data = df_userPlans_with_click[df_userPlans_with_click['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Distance', 'ETA']]
            data_transportMode_label_train[index] = temp_data['isSelected']
        elif index == 7:
            #for transportMode == 8, dismiss the price
            temp_data = df_userPlans_with_click[df_userPlans_with_click['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'ETA']]
            data_transportMode_label_train[index] = temp_data['isSelected']
        else:
            temp_data = df_userPlans_with_click[df_userPlans_with_click['TransportMode'] == (index + 1)]
            data_transportMode_train[index] = temp_data[['Order', 'Hour', 'Distance', 'Price', 'ETA']]
            data_transportMode_label_train[index] = temp_data['isSelected']

    #dump to files
    pickle.dump(data_transportMode_train, open('./data_set_phase1/train_data/all_data_'+ str(train_index) + '.pkl', 'wb'))
    pickle.dump(data_transportMode_label_train, open('./data_set_phase1/train_data/all_label_'+ str(train_index) + '.pkl', 'wb'))


def train(train_index):
    print("train model");
    models = [GradientBoostingClassifier() for i in range(11)]
    data = pickle.load(open('./data_set_phase1/train_data/all_data_'+ str(train_index) + '.pkl', 'rb'))
    label = pickle.load(open('./data_set_phase1/train_data/all_label_'+ str(train_index) + '.pkl', 'rb'))
    for i in range(11):
        print(i)
        models[i].fit(data[i], label[i])

    for i in range(11):
        pickle.dump(models[i], open('./data_set_phase1/model/Xgboost/all_model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'wb'))


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
        models.append(pickle.load(open('./data_set_phase1/model/Xgboost/all_model_for_transport_'+ str(train_index) + '_%d.pkl' % i, 'rb')))

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
                plan_data = [eplan[['Order', 'Hour', 'Distance', 'ETA']]]
            elif eplan['TransportMode'] == 4:
                plan_data = [eplan[['Order', 'Distance', 'ETA']]]
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

    data_gen(hour_m3, 1)
    train(1)
    get_result(hour_m3, 1)


    # get_result()