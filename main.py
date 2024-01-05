import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

## NEED PARSING
# === PARSING === #

# === GET DATAS AND VISUALIZE === #
def get_data_values(data):
	return data.iloc[:, 0], data.iloc[:, 1]

def scatter_dataset(X, Y):
	try:
		plt.scatter(X, Y, label='value', color='blue', marker='o')

		plt.xlabel('Kms')
		plt.ylabel('Price')
		plt.title('data.csv representation')

		plt.show()
	except e as Exception:
		print(F"{e}")

def plot_predictions(X, Y, Y_pred):
	try:
		plt.clf()
		plt.scatter(X, Y, label='Data Points', color='blue', marker='o')
		# Fit a line through the points
		plt.plot([min(X), max(Y_pred)], [max(X), min(Y_pred)], label='Line', color='green', linestyle='--', marker='o')
		# Add labels and title
		plt.xlabel('Kms')
		plt.ylabel('Price')
		plt.title('data.csv representation')
		# Show the plot
		plt.show()
	except e as Exception:
		print(f"{e}")

# === MATHS FUNCTIONS === #
def get_Yprediction(m, X, b):
	return m * X + b

def mean_square_error(X, Y, n, m, b):
	return sum((Y - (m * X + b)) ** 2) / n

def gradient_descent(X, Y, n, m, b):
	L = 0.001
	clip_value = 5.0  # Set a threshold for gradient clipping
	Y_pred = m * X + b
	Dm = (-2 / n) * sum(X * (Y - Y_pred))
	Db = (-2 / n) * sum(Y - Y_pred)
	Dm = max(min(Dm, clip_value), -clip_value)
	Db = max(min(Db, clip_value), -clip_value)
	m = m - L * Dm
	b = b - L * Db
	return m, b

# === MAIN === #
if __name__ == "__main__":
	data = pd.read_csv('./DataSet/data.csv')

	X, Y = get_data_values(data)

	scatter_dataset(X, Y)


	# initialize parameters
	m = 0 # slope
	b = 0 # intercept
	n = len(X) # n values in dataset
	L = 0.01
	epochs = 4000

	costs = [0] * 4000

	# calcul cost before training (diff between prediction and real value)
	# print(f"cost before training: {cost}")

	# gradient descent
	# iterate
	for i in range(epochs):
		costs[i] = mean_square_error(X, Y, n, m, b)
		m, b = gradient_descent(X, Y, n, m, b)

	# make predictons
	# print(f"prediction: {m * 7000 + b}")

	# # evaluate

	# # visualize
	plot_predictions(X, Y, get_Yprediction(m, X, b))
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('Cost Function over Iterations')
	plt.plot(np.arange(epochs), costs, label='Cost Function')
	plt.show()