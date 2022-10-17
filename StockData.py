import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Stock:
	def __init__(self, ticker, start_date, end_date):
		self.ticker = ticker
		self.start = start_date
		self.end = end_date
		self.stock_data = self.get_data()

	def get_data(self):
		start = pd.to_datetime([self.start]).astype(int)[0]//10**9
		end = pd.to_datetime([self.end]).astype(int)[0]//10**9
		url = 'https://query1.finance.yahoo.com/v7/finance/download/' + self.ticker + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
		df = pd.read_csv(url, parse_dates=True, index_col=0)
		return df['Close']

	def plot(self):
		self.stock_data.plot(label=self.ticker)
		plt.ylabel(f'{self.ticker} Close')
		plt.grid()
		plt.legend()
		plt.show()
