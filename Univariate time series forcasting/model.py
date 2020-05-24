from time_series_data_gen import TimeSeriesDataGenerator
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt 
import numpy as np 

datagen = TimeSeriesDataGenerator()

time_series_data = [110, 125, 133, 146, 158, 172, 187, 196, 210]

n_steps = 3
X, y = datagen.prepare_from_given_data(time_series_data, n_steps) 

n_features = 1
X = X.reshape(X.shape[0], X.shape[1], n_features) #(samples, timesteps, n_features)

# RNN Model
model = Sequential()
model.add(LSTM(units=64, activation="relu", return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(units=64, activation="relu"))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=300)
model.save("univariate_time_series.h5")
