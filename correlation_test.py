import numpy as np
import pandas as pd
#import pandas.io.data as web
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime 

start = datetime.datetime(2000,1,1)
end = datetime.datetime(2016,5,20)
plusDay = datetime.timedelta(days=1)

stocks = ["^GSPC", "AAPL", "GE", "PG", "INTC", "TM", "KO", "JPM", "T"]
correlation_factors = ["^BSESN", "^FCHI", "^FTSE", "CNY=X", "JPY=X", "GBP=X", "EUR=X"] #India, France, England, USD/CNH, USD/JPY, USD/GBP, USD/EUR

# Stocks to check correlation with
stock_data = web.DataReader(stocks, 'yahoo', start, end)["Adj Close"]
# Potential features
factors = web.DataReader(correlation_factors, 'yahoo', start, end)["Adj Close"]

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

