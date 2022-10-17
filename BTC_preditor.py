import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler 
from StockData import Stock

stock = Stock('BTC-USD', '2015-1-1', '2022-10-17')
data = stock.get_data()
scaler = MinMaxScaler(feature_range=(0, 1))

def scale(x):
	scaled = scaler.fit_transform(x.reshape(-1, 1))
	return scaled
def inverse(x):
	return scaler.inverse_transform(x.reshape(-1, 1))

train_data = data.loc['2015':'2021']
test_data = data.loc['2022']

def prepare_data(data):
    scaled_close = scale(data.values) 
    X = []
    Y = []
    
    for x in range(50, len(scaled_close)):
        X.append(scaled_close[x-50:x, 0])
        Y.append(scaled_close[x, 0])
    
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1],  1))
    Y = np.array(Y)
    
    return X, Y

x_train, y_train = prepare_data(train_data)
x_test, y_test = prepare_data(test_data)

model = Sequential()
model.add(LSTM(units=40, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(units=40, return_sequences=True))
model.add(LSTM(units=40))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size=32)
model.save('btc-predictor')
#model = load_model('btc-predictor')

def testing(x, y, nn):
	yhat = nn.predict(x)

	fig, ax = plt.subplots()
	ax.plot(y, label='True Close')
	ax.plot(yhat, label='Machine Prediction')
	ax.legend()
	ax.grid()
	plt.show()

#This function predicts (single_pred) from the last 50 values from the data "close" and and pushes (single_pred) to the "close"
def future_predictor(close):
  last_close = close[len(close)-50 :]
  last_close = scale(last_close)
  last_close = np.reshape(last_close, (1, len(last_close), 1))
  single_pred = model.predict(last_close)
  single_pred = inverse(single_pred)

  new_close = np.concatenate((close, np.reshape(single_pred, (1))))
  return new_close

#This function iterates over given days and returns the future pred : len(pred) = days + 1
def future_prediction(days):
	close = data.values
	for day in range(days):
	  close = future_predictor(close)

	btc_pred = close[len(close) - (days + 1):] # we add 1 to return the last value in our data with our pred
	return btc_pred

testing(x_test, y_test, model)
prediction = future_prediction()
