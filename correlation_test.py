import numpy as np
import pandas as pd
#import pandas.io.data as web
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2016,5,20)
day = datetime.timedelta(days=1)

stocks = ["^GSPC", "AAPL", "GE", "PG", "INTC", "TM", "KO", "JPM", "T"] #The stocks to check correlation with.

correlation_factors_no_offset = ["^SSEC", "^N225", "^BSESN"] #China, Japan, India
correlation_factors_offset = ["^DJI", "^IXIC", "^GDAXI", "^FCHI", "^FTSE", "CNY=X", "JPY=X", "GBP=X", "EUR=X"] #DOW, Nasdaq, Germany, France, England, USD/CNY, USD/JPY, USD/GBP, USD/EUR
correlation_factors = correlation_factors_no_offset+correlation_factors_offset

stock_data = web.DataReader(stocks, 'yahoo', start, end)["Adj Close"]
print("Collected Stock Data")

factors_without_offset = web.DataReader(correlation_factors_no_offset, 'yahoo', start, end)["Adj Close"]
factors_with_offset = web.DataReader(correlation_factors_offset, 'yahoo', start-day, end-day)["Adj Close"]
factors_with_offset = factors_with_offset.shift(1)

factors = pd.concat([factors_without_offset, factors_with_offset], axis=1)

for stock in stocks:
	for factor in correlation_factors:
		correlation_table = pd.concat([stock_data[stock], factors[factor]], axis=1)
		correlation_table = correlation_table.dropna()

		print("\n\n" + factor + " - " + stock)
		print(correlation_table[stock].corr(correlation_table[factor]))
		print(correlation_table)
		correlation_table.plot.scatter(x=stock, y=factor)
		plt.show()

# print(factors)
# print(len(factors.index))
# num_rows = len(stockRawData.index)
# print(num_rows)