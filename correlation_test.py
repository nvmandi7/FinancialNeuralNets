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

# Consider changing to returns?


corrs = np.ndarray((len(correlation_factors), len(stocks)))
for i in range(len(stocks)):
	for j in range(len(correlation_factors)):
		factor = correlation_factors[j]
		stock = stocks[i]

		correlation_table = pd.concat([stock_data[stock], factors[factor]], axis=1)
		correlation_table = correlation_table.dropna()

		print("\n\n" + factor + " - " + stock)
		corrs[j][i] = correlation_table[stock].corr(correlation_table[factor])
		print(corrs[j][i])
		# print(correlation_table)
		# correlation_table.plot.scatter(x=stock, y=factor)
		# plt.show()
data = np.hstack((np.array(correlation_factors).reshape((len(correlation_factors), 1)), corrs.astype(str)))
np.savetxt('corr_table.csv', data, fmt='%s', delimiter=',', newline='\n', header='Stock, ' + ", ".join(stocks))


# print(factors)
# print(len(factors.index))
# num_rows = len(stockRawData.index)
# print(num_rows)