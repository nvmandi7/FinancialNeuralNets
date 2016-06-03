from retrieve_data import *
from scipy.stats import pearsonr
print("retrieve_data finished")


plusDay = datetime.timedelta(days=1)
test = create_lookback_returns_data(web.DataReader("^GSPC", 'yahoo', start+plusDay, end+plusDay), 1)

tickers = ['s&p', 'dow', 'nasdaq', 'china', 'japan', 'germany', ]

snp = create_lookback_returns_data(web.DataReader("^GSPC", 'yahoo', start, end), 1)
dow = create_lookback_returns_data(web.DataReader("^DJI", 'yahoo', start, end), 1)
nas = create_lookback_returns_data(web.DataReader("^IXIC", 'yahoo', start, end), 1)
# dow = create_lookback_returns_data(web.DataReader("^DJI", 'yahoo', start, end), 1)








