from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

def getExtra(df):
    conditions = [ \
        (df['overnight'] == True) & (df['rush_hour'] == False).isin([1,3,4]), \
        (df['overnight'] == True) & (df['rush_hour'] == True).isin([1,3,4]), \
        (df['overnight'] == False) & (df['rush_hour'] == True) & (df['RatecodeID'] == 2), \
        (df['overnight'] == False) & (df['rush_hour'] == True) & (df['RatecodeID'].isin([1,3,4])), \
    ]
    choices = [.5, 1.5, 4.5, 1]
    return np.select(conditions, choices, default=0)

def getHolidays(df):
    # Get Holidays
    cal = calendar()
    dr = pd.date_range(start=df['tpep_pickup_datetime'].min(), end=df['tpep_pickup_datetime'].max())
    holidays = cal.holidays(start=dr.min(), end=dr.max())
    return df['tpep_pickup_datetime'].isin(holidays)
    
def getOvernight(df):
    return (pd.to_datetime(df.tpep_pickup_datetime).dt.hour <= 5) | \
    (pd.to_datetime(df.tpep_pickup_datetime).dt.hour >= 20)
    
def getRushHour(df):
    return ((pd.to_datetime(df.tpep_pickup_datetime).dt.hour < 20) & \
      (pd.to_datetime(df.tpep_pickup_datetime).dt.hour >= 16)) \
    & \
    (df['day'].isin(["Monday","Tuesday","Wednesday","Thursday","Friday"])) &\
    (df['holiday'] == False)
    

def evaluateLGB(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    r2 = r2_score(test_labels,predictions)
    mea = mean_absolute_error(test_labels,predictions)
    err = pd.DataFrame(test_labels-predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('R2 Score: {:0.4f}'.format(r2))
    print('MAE: {:0.4f}'.format(mea))
    
    sns.distplot(err,hist_kws={"log":True},kde=False);
    plt.show()
    sns.regplot(x=predictions, y=test_labels)
    plt.show()
    lgb.plot_importance(model.best_estimator_, max_num_features=10)
    plt.show()
    
    
def evaluateRF(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    r2 = r2_score(test_labels,predictions)
    mea = mean_absolute_error(test_labels,predictions)
    err = pd.DataFrame(test_labels-predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('R2 Score: {:0.4f}'.format(r2))
    print('MAE: {:0.4f}'.format(mea))
        
    sns.distplot(err,hist_kws={"log":True},kde=False);
    plt.show()
    sns.regplot(x=predictions, y=test_labels)
    plt.show()
    
def evaluateSVM(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    r2 = r2_score(test_labels,predictions)
    mea = mean_absolute_error(test_labels,predictions)
    err = pd.DataFrame(test_labels-predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('R2 Score: {:0.4f}'.format(r2))
    print('MAE: {:0.4f}'.format(mea))
        
    sns.distplot(err,hist_kws={"log":True},kde=False);
    plt.show()