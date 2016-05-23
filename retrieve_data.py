import numpy as np
import pandas as pd
#import pandas.io.data as web
import pandas_datareader.data as web
from datetime import datetime

start = datetime(2000,1,1)
end = datetime(2016,5,20)

#S&P Prediction
stockRawData = web.DataReader("^GSPC", 'yahoo', start, end)
num_rows = len(stockRawData.index)

def create_lookback_returns_data(stock_data, num_days):
	num_rows = len(stock_data.index)
	stock_data_features = np.zeros([num_rows-num_days+1,num_days])
	for day in range(0, num_rows-num_days+1):
		for i in range(num_days):
			new_day = day+i
			adjustment_factor = float(stock_data['Adj Close'][new_day]/stock_data['Close'][new_day])
			adjusted_open = adjustment_factor*stock_data['Open'][new_day]
			stock_data_features[day,i] = 100.0*(stock_data['Adj Close'][new_day]-adjusted_open)/adjusted_open
	return stock_data_features

lookback_days = 5 #includes current day
stock_data = create_lookback_returns_data(stockRawData, lookback_days)
stock_data = np.delete(stock_data, -1, 0) #Remove last row since we can't label it

'''
Creating classification labels. 
BUY or 1 if returns for next day are positive
SELL or 0 otherwise
'''
def create_labels(stock_data, lookback_days):
	num_rows = len(stock_data.index)
	labels = []
	for i in range(lookback_days-1, num_rows-1):
		cur_day = stock_data.iloc[i+1]['Adj Close']
		prev_day = stock_data.iloc[i]['Adj Close']
		day_return  = 100.0*(cur_day-prev_day)/prev_day
		if day_return < 0:
			labels.append(0) #SELL
			# stock_data.iloc[[i]] = -stock_data.iloc[[i]]
		else:
			labels.append(1) #BUY
	return labels

labels = create_labels(stockRawData, lookback_days)


# combined = pd.concat([stockRawData,googStockData],axis=1)
# stockRawData = stockRawData.drop(stockRawData.index[-1:])
# stockRawData = stockRawData.iloc[np.random.permutation(len(stockRawData))]
# data_to_process = stockRawData.values

'''
Normalizing the data using
MAX-MIN Normalization in range [-1,1]
'''
def normalize_data(data):
	num_cols = data.shape[1]
	for i in range(num_cols):
		col = data[:,i]
		minimum = col.min()
		maximum = col.max()
		rangeOfCol = maximum-minimum
		col = 2*(col-minimum)/(rangeOfCol)-1
		data[:,i] = col
	return data

normalized_data = normalize_data(stock_data)

