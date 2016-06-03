import numpy as np
import pandas as pd
#import pandas.io.data as web
import pandas_datareader.data as web
import datetime
from scipy.stats import zscore

# ------------------------------------------
# Time Series Features

start = datetime.datetime(2004,1,1)
end = datetime.datetime(2016,5,20)
day = datetime.timedelta(days=1)

#S&P Prediction
# stockRawData = web.DataReader("^GSPC", 'yahoo', start, end)

correlation_factors_no_offset = ["^SSEC", "^N225", "^BSESN"] #China, Japan, India
correlation_factors_offset = ["^GSPC", "^DJI", "^IXIC", "^GDAXI", "^FCHI", "^FTSE", "CNY=X", "JPY=X", "GBP=X", "EUR=X"] #DOW, Nasdaq, Germany, France, England, USD/CNY, USD/JPY, USD/GBP, USD/EUR
correlation_factors = correlation_factors_no_offset+correlation_factors_offset
factors_without_offset = web.DataReader(correlation_factors_no_offset, 'yahoo', start, end)["Adj Close"]
factors_with_offset = web.DataReader(correlation_factors_offset, 'yahoo', start-day, end-day)["Adj Close"]
factors_with_offset = factors_with_offset.shift(1)
factors = pd.concat([factors_without_offset, factors_with_offset], axis=1)
factors = factors.diff()/factors.shift(1)*100

# num_rows = len(stockRawData.index)

def create_lookback_returns_data(stock_adj_close, sample_close, sample_open, num_days):
	num_rows = len(stock_adj_close.index)
	stock_data_features = np.zeros([num_rows-num_days+1,num_days])
	for day in range(0, num_rows-num_days+1):
		for i in range(num_days):
			new_day = day+i
			adjustment_factor = float(stock_adj_close[new_day]/sample_close[new_day])
			adjusted_open = adjustment_factor*sample_open[new_day]
			stock_data_features[day,i] = 100.0*(stock_adj_close[new_day]-adjusted_open)/adjusted_open
	return stock_data_features

lookback_days = 10 #includes current day
# stock_data = create_lookback_returns_data(stockRawData, lookback_days)
# stock_data = np.delete(stock_data, -1, 0) #Remove last row since we can't label it

'''
Creating classification labels. 
BUY or 1 if returns for next day are positive
SELL or 0 otherwise
'''
def create_labels(stock_adj_close, lookback_days):
	hold = 1
	num_rows = len(stock_adj_close.index)
	labels = []
	for i in range(lookback_days-1, num_rows-1):
		cur_day = stock_adj_close.iloc[i+1]
		prev_day = stock_adj_close.iloc[i]
		day_return  = 100.0*(cur_day-prev_day)/prev_day
		if day_return < -hold:
			labels.append(0) #SELL
			# stock_data.iloc[[i]] = -stock_data.iloc[[i]]
		elif day_return > hold:
			labels.append(2) #BUY
		else:
			labels.append(1)
	return labels

# labels = create_labels(stockRawData, lookback_days)


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

# normalized_data = normalize_data(stock_data)

sample = 'AAPL'
sampleRawData = web.DataReader(sample, 'yahoo', start, end)


# ------------------------------------------
# Other Features

correlation_factors_no_offset = ["^SSEC", "^N225", "^BSESN"] #China, Japan, India
correlation_factors_offset = ["^GSPC", "^DJI", "^IXIC", "^GDAXI", "^FCHI", "^FTSE", "CNY=X", "JPY=X", "GBP=X", "EUR=X"] #DOW, Nasdaq, Germany, France, England, USD/CNY, USD/JPY, USD/GBP, USD/EUR
correlation_factors = correlation_factors_no_offset+correlation_factors_offset


factors_without_offset = web.DataReader(correlation_factors_no_offset, 'yahoo', start, end)["Adj Close"]
factors_with_offset = web.DataReader(correlation_factors_offset, 'yahoo', start-day, end-day)["Adj Close"].shift(1)
	

factors = pd.concat([sampleRawData['Adj Close'], sampleRawData['Open'], sampleRawData['Close'], factors_without_offset, factors_with_offset], axis=1).dropna()

# ------------------------------------------
# Frequency Domain Features
sampleAdjCloseData = factors['Adj Close']
sampleCloseData = factors['Close']
sampleOpenData = factors['Open']
sampleDataTime = create_lookback_returns_data(sampleAdjCloseData, sampleCloseData, sampleOpenData, lookback_days)
sampleDataFreq = np.absolute(np.fft.fft(sampleDataTime))

###
# Design matrix
factors = 100.0 * factors.diff() / factors.shift(1)
factors = factors.drop(['Adj Close', 'Open', 'Close'], 1)
factors = factors.ix[lookback_days-1:]

design = np.hstack([factors, sampleDataTime, sampleDataFreq])[:-1]
design = zscore(design)
labels = create_labels(sampleAdjCloseData, lookback_days)











