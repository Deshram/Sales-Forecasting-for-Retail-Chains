import pandas as pd
import numpy as np
import pickle
from flask import Flask, jsonify, request

import flask
app = Flask(__name__)

def fe(item_id, store_id, sales, cal, price):
    
    data_point = sales[(sales['item_id'] == item_id) & (sales['store_id'] == store_id)].values
    data_point = pd.DataFrame(data_point,columns = sales.columns)
    d_cols = [d for d in sales.columns if 'd_' in d]
    data = data_point.drop(d_cols[:1883],axis = 1)
    
    #making cols for days 1942-69 amd filling it with zero
    for day in range(1942,1970):
        data['d_' + str(day)] = 0
        data['d_' + str(day)] = data['d_' + str(day)].astype(np.int16)
    
    pre_data = pd.melt(data, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                      var_name='d', value_name='sales')
    
    #combining the dataset 
    pre_data = pd.merge(pre_data, cal, on='d', how='left')
    pre_data = pd.merge(pre_data, price, on=['store_id','item_id','wm_yr_wk'], how='left')
    
    #fil the missing sell price values by mean imputaion
    pre_data["sell_price"].fillna(pre_data.groupby("id")["sell_price"].transform("mean"), inplace=True)
    
    pre_data.drop(columns=["date","weekday"], inplace=True)
    pre_data['d'] = pre_data['d'].apply(lambda a: a.split('_')[1]).astype(np.int16)

    #calculating lags feature
    lags = [1,2,3,5,7,14,21,28]
    for lag in lags:
        pre_data["lag_" + str(lag)] = pre_data.groupby("id")["sales"].shift(lag).astype(np.float16)

    #calculating rolling features
    pre_data['rolling_mean_10'] = pre_data.groupby("id")['sales'].transform(lambda x: x.rolling(10).mean())
    pre_data['rolling_mean_20'] = pre_data.groupby("id")['sales'].transform(lambda x: x.rolling(20).mean())
    pre_data['rolling_mean_30'] = pre_data.groupby("id")['sales'].transform(lambda x: x.rolling(30).mean())

    #for query data points keeping lag and rolling features same as previous 28 days lag and rolling features 
    pre_data.iloc[1941:,-11:] = pre_data.iloc[1913:1941,-11:].values
    pre_data = pre_data[pre_data['d'] >= 1942]
    pre_data.drop(['id', 'd','sales', 'wm_yr_wk'],axis = 1,inplace = True)

    cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1','event_type_1', 'event_name_2', 'event_type_2']

    for i in cat_cols:
        if i in sales.columns:
            dict_id = dict(enumerate(sales[i].cat.categories))
        else:
            dict_id = dict(enumerate(cal[i].cat.categories))
        
        keys = list(dict_id.keys())
        values = list(dict_id.values())
        og_values = list(pre_data[i].unique())
        replace_values = []
        for j in og_values:
            if j == 'No_event':
                replace_values.append(-1)
            else:
                replace_values.append(keys[values.index(j)])

        pre_data[i].replace(og_values,replace_values,inplace = True)        
            
    model_file = open('cgb.pkl', 'rb')
    model = pickle.load(model_file)

    forecast_values = model.predict(pre_data)

    return np.rint(forecast_values)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	sales = pd.read_pickle("sales_ad.pkl")
	cal = pd.read_pickle("cal_ad.pkl")
	price = pd.read_pickle("prices_ad.pkl")
	inp = request.form.to_dict()
	forecast = fe(inp['Item id'], inp['store_id'], sales, cal, price)
	return {'Further 28 days forecast': forecast.tolist()}

if __name__ == '__main__':
    app.run()

