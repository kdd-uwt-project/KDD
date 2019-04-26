import pickle
import pandas as pd

user_click_file = open('./data_set_phase1/train_clicks.pickle',"rb")
user_click = pickle.load(user_click_file)
user_click_file.close()

user_plans_file = open('./data_set_phase1/train_plans.pickle', "rb")
user_plans = pickle.load(user_plans_file)
user_plans_file.close()

user_queries_file = open('./data_set_phase1/train_queries.pickle', "rb")
user_queries = pickle.load(user_queries_file)
user_queries_file.close()

#convert to dataframe
df_userPlans = pd.DataFrame(user_plans, columns=['Sid','PlanTime','Distance','Price','ETA','TransportMode'])
df_userClick = pd.DataFrame(user_click, columns = ['Sid','ClickTime', 'TransportMode'])
df_userClick = pd.merge(df_userClick, df_userPlans,
                      how='left', left_on=['Sid','TransportMode'], right_on=['Sid','TransportMode']).drop_duplicates(subset=['Sid'], keep=False)