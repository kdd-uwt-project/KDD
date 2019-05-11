import sklearn
import pickle
from sklearn.svm import SVC

def data_gen():
    train_plans = pickle.load(open('./data_set_phase1/train_plans.pickle', 'rb'))
    train_clicks = pickle.load(open('./data_set_phase1/train_clicks.pickle', 'rb'))

    train_plans.sort(key=lambda x: x[0])
    train_clicks.sort(key=lambda x: x[0])

    print(train_plans[0])
    print(train_clicks[0])

    data = [[[] for i in range(11)], [[] for i in range(11)]]
    data_label = [[[] for i in range(11)], [[] for i in range(11)]]
    index = 0

    count = 0
    click_index = 0
    for plan in train_plans:
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
        for eplan in plan[2]:
            data[index][eplan['transport_mode']-1].append([hour, eplan['distance'], eplan['price'] if eplan['price'] != '' else 0, eplan['eta']])
            data_label[index][eplan['transport_mode']-1].append(1 if target_mode == eplan['transport_mode'] else 0)
        count += 1
        if count > len(train_plans) / 5 * 4:
            index = 1

    pickle.dump(data[0], open('./train_data/data.pkl', 'wb'))
    pickle.dump(data[1], open('./test_data/data.pkl', 'wb'))
    pickle.dump(data_label[0], open('./train_data/label.pkl', 'wb'))
    pickle.dump(data_label[1], open('./test_data/label.pkl', 'wb'))


def train():
    models = [SVC(probability=True) for i in range(11)]
    data = pickle.load(open('./train_data/data.pkl', 'rb'))
    label = pickle.load(open('./train_data/label.pkl', 'rb'))
    for i in range(11):
        print(i)
        models[i].fit(data[i][:len(data[i]) // 10], label[i][:len(data[i]) // 10])

    for i in range(11):
        pickle.dump(models[i], open('./model/model_for_transport_%d.pkl' % i, 'wb'))


def test():
    models = []
    data = pickle.load(open('./test_data/data.pkl', 'rb'))
    label = pickle.load(open('./test_data/label.pkl', 'rb'))
    for i in range(11):
        models.append(pickle.load(open('./model/model_for_transport_%d.pkl' % i, 'rb')))

def get_result():
    data = pickle.load(open('./data_set_phase1/test_plans.pickle', 'rb'))
    models = []
    f = open("svm.csv", 'w')
    f.write('"sid","recommend_mode"\n')
    for i in range(11):
        models.append(pickle.load(open('./model/model_for_transport_%d.pkl' % i, 'rb')))
    count = 0
    for query in data:
        # print(len(query[2]))
        max_value = 0
        max_index = 0
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
            temp = [[hour, plan['distance'], plan['price'] if plan['price'] != '' else 0, plan['eta']]]

            result = models[plan['transport_mode']-1].predict_proba(temp)[0]
            if result[1] > max_value:
                max_value = result[1]
                max_index = plan['transport_mode']
        count += 1
        if count % 1000 == 999:
            print(count)

        if max_value > 0.1:
            f.write('"%d","%d"\n' % (query[0], max_index))
        else:
            f.write('"%d","%d"\n' % (query[0], 0))


if __name__ == '__main__':
    # train()
    get_result()