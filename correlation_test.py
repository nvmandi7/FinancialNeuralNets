import numpy as np
import pandas as pd
#import pandas.io.data as web
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime

start = datetime(2000,1,1)
end = datetime(2016,5,20)

stocks = ["^GSPC", "AAPL", "GE", "PG", "INTC", "TM", "KO", "JPM", "T"]

correlation_factors = ["^BSESN", "^FCHI", "^FTSE", "CNY=X", "JPY=X", "GBP=X", "EUR=X"] #India, France, England, USD/CNH, USD/JPY, USD/GBP, USD/EUR

#The stocks to check correlation with.
stock_data = web.DataReader(stocks, 'yahoo', start, end)["Adj Close"]
print("Collected Stock Data")

factors = web.DataReader(correlation_factors, 'yahoo', start, end)["Adj Close"]

for stock in stocks:
	for factor in correlation_factors:
		correlation_table = pd.concat([stock_data[stock], factors[factor]], axis=1)
		correlation_table = correlation_table.dropna()

		print("\n\n" + factor + " - " + stock)
		print(correlation_table[stock].corr(correlation_table[factor]))
		# print(correlation_table)
		# correlation_table.plot.scatter(x=stock, y=factor)
		# plt.show()

# print(factors)
# print(len(factors.index))
# num_rows = len(stockRawData.index)
# print(num_rows)

