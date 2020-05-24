import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Flatten, Activation
import matplotlib.pyplot as plt 
import numpy as np 

model = load_model("univariate_time_series.h5")

# predicting for next 10 days
input_X = np.array([187, 196, 210])

curr_input = list(input_X)
curr_output = []

n_steps = 3
n_features = 1

i = 0
while i < 10:
	if len(curr_input) > 3:
		input_X = np.array(curr_input[1:])
		print("{} day input: {}".format(i, input_X))
		input_X = input_X.reshape(1, n_steps, n_features)
		y_pred = model.predict(input_X)[0]
		print("{} day output: {}".format(i, y_pred))
		curr_input.append(y_pred[0])
		curr_input = curr_input[1:]
		curr_output.append(y_pred[0])
		i += 1
	else:
		input_X = input_X.reshape(1, n_steps, n_features)
		y_pred = model.predict(input_X)[0]
		print("1st output: ", y_pred[0])
		curr_input.append(y_pred[0])
		curr_output.append(y_pred[0])
		i += 1

# Visualising predictions:
current_data_points = np.arange(1,10)
next_10_data_predictions = np.arange(10,20)

plt.plot(current_data_points, [110, 125, 133, 146, 158, 172, 187, 196, 210])
plt.plot(next_10_data_predictions, curr_output)
plt.savefig("predictions.png")
plt.show()