from retrieve_data import sampleAdjCloseData
from finance_nn import test_labels, test_days
import numpy as np
from matplotlib import pyplot as plt


all_returns = sampleAdjCloseData.diff() / sampleAdjCloseData.shift(1)
returns = np.array(all_returns[-test_days:])
initial = 1000 # Money to start with


# All-in Strategy - No shorting
money = []
current = initial
isBought = False
for day in range(test_days):
	if test_labels[day] == 2:
		current *= (1 + returns[day])
		isBought = True
	elif test_labels[day] == 0:
		isBought == False
	elif isBought and test_labels[day] == 1:
		current *= (1 + returns[day])
	money.append(current)

print(money)
market_diff = sampleAdjCloseData[-1] - sampleAdjCloseData[-test_days]
benchmark = initial * (1+market_diff/sampleAdjCloseData[-test_days])
print(benchmark)
print(min(money), max(money))

plt.plot(money)
plt.plot(sampleAdjCloseData[-test_days:]/sampleAdjCloseData[-test_days] * 1000)
plt.show()