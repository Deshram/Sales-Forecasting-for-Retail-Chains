# M5_accuracy

Forecasting sales for further 28 days for a given item of a store<br/>
Dataset: https://www.kaggle.com/c/m5-forecasting-accuracy/data<br/>
Copetition overview: https://www.kaggle.com/c/m5-forecasting-accuracy/overview<br/>

Dataset Overview:<br/>
![alt text](https://github.com/Deshram/M5_accuracy/blob/main/screenshots/Dataset_overview.jpg)

<br>The dataset consist of sales of previous 1941 days sales of 3049 items in 10 stores of 3 states in US. Apart from historical sales data we also have rate of each item at corresponding store and dates information like events on that corresponding date.</br>

<br>A customized metric known as WRMSSE based on MAPE is used as performance metric. 

<br>Performed 4 models on the dataset (Simpel Exponential Smoothing, XGBoostRegressor, CatBoostRegressor, LGBMRegressor).</br>

EDA_FE: Performed preprocessing and Exploratory Data Analysis on dataset and introduced lags and rolling features. Converted time series problem to supervised machine learning problem.<br/>
ses: Performed simple Exponential smoothing.<br/>
models: Performed all three above mentioned bossting algorithms.<br/>
final: Final deployment model.<br/>

Scores:<br/>
![alt text](https://github.com/Deshram/M5_accuracy/blob/main/screenshots/scores.JPG)

Choosed CatBoostRegressor for final model.

Out of 5558 participants the ranks for score 0.685 were in range of 490-500's i.e the score can be considered as top 10% percentile rank. 
