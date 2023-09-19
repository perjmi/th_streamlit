"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd
import random

instrument="espidxeur"

def is_market_open(dt: datetime) -> bool:
    dt = dt.astimezone(pytz.timezone('Europe/London'))
    #return 0 <= dt.weekday() <= 4 and 9*60+30 <= dt.hour*60+dt.minute <= 16*60
    return 0 <= dt.weekday() <= 4 and 8*60 <= dt.hour*60+dt.minute < 16*60+30
    
#write a function that aggregates ohlc data in a pandas dataframe

def aggregate_ohlc(df,freq='1min'):
    dfc=df.copy()
    dfc.set_index('utc_time', inplace=True)
    dfc = dfc.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_no_nan = dfc.dropna()
    return df_no_nan

@st.cache_data
def load_data():
    data=pd.read_csv("espidxeur-m5-bid-2014-06-02-2023-09-10.csv")

    data['utc_time'] = pd.to_datetime(data['timestamp'], unit='ms').dt.tz_localize('UTC')
    data['utc_time'] = data['utc_time'].dt.tz_convert('Europe/London')

    data_open = data[data['utc_time'].apply(lambda x: is_market_open(x))]

    data_5min=aggregate_ohlc(data_open,freq='5min')

    row_count=data_5min.resample('1D').size()
    dates=row_count[row_count==102].index
    ind=[ a in set(dates.date) for a in data_5min.index.date ]
    data_5min_full=data_5min.copy()[ind]
    return data_5min_full

data_5min_full=load_data()

if "j" not in st.session_state:
    st.session_state.j = 0

if "i" not in st.session_state:
    st.session_state.i = 24

forward_clicked = st.button('Forward one day')
forward_clicked_bar = st.button('Forward one bar')
backward_clicked_bar = st.button('Backward one bar')

if forward_clicked:
    st.session_state.j += 102
    st.session_state.i = 6

if forward_clicked_bar:
    st.session_state.i += 1

if backward_clicked_bar:
    st.session_state.i -= 1


j=st.session_state.j
i=st.session_state.i

#j=102*random.randint(0,1000)
#df0=data_5min_full.iloc[j:(j+102)]
df0=data_5min_full.iloc[j:(j+51)]
df=(df0.iloc[:i])

low_point=[(df.index[2],df['low'].iloc[3]),(df.index[4],df['low'].iloc[3])]
up_point=[(df.index[2],df['high'].iloc[3]),(df.index[4],df['high'].iloc[3])]
#fig,axlist=mpf.plot(df,type='candle',style='charles',alines=[low_point,up_point],returnfig=True)
fig,axlist=mpf.plot(df,type='candle',alines=[low_point,up_point],returnfig=True,xlim=(df0.index[0]-(df0.index[1]-df0.index[0]),df0.index[-1]+(df0.index[1]-df0.index[0])),ylim=(df['low'].min()-10,df['high'].max()+10),title=instrument+" "+df.index[0].strftime("%Y/%m/%d"),figscale=1.4)

#apdict = mpf.make_addplot(df.iloc[2]['High']+30)

#axlist[0].text(df.index[50], df['Close'][50], 'hi mplfinance')
#axlist[0].text(0.5, 0.5, 'hi mplfinance', horizontalalignment='center', verticalalignment='center', transform=axlist[0].transAxes)

#mpf.plot(df, type='candle', style='yahoo')
#fig = plt.gcf()
#plt.tifle("My title")
#plt.show()

#ax.text(x=20, y=1,s='Text Here')
yrange=df['high'].max()-df['low'].min()
yslack=yrange/25

x = 2
y = df.loc[df.index[3],'high']+yslack
axlist[0].text(x,y,round(df.iloc[3]['high']))

x = 2
y = df.loc[df.index[3],'low']-yslack
axlist[0].text(x,y,round(df.iloc[3]['low']))


max_indices = argrelextrema(df['high'].values, np.greater,order=2)[0]
for x in max_indices:
    if x>3:
        y = df.loc[df.index[x],'high']+yslack
        axlist[0].text(x,y,round(df.iloc[x]['high']))

min_indices = argrelextrema(df['low'].values, np.less,order=2)[0]
for x in min_indices:
    if x>3:
        y = df.loc[df.index[x],'low']-yslack
        axlist[0].text(x,y,round(df.iloc[x]['low']))
st.pyplot(fig)
st.write(i)

