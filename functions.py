from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import pandas as pd

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
    