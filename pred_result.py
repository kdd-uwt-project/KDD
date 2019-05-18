import pickle


pred = pickle.load(open('test_pred_result.pkl'))
label = pickle.load(open('./test_data/label.pkl', 'rb'))

for p in pred:
    if p == 0:
        print(p)