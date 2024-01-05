import pandas as pd
import matplotlib.pyplot as plt
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
	epochs = 10000
	prev_cost = mean_square_error(X, Y, n, m, b)
	for i in range(epochs):
		Y_pred = m * X + b
		Dm = (-2 / n) * sum(X * (Y - Y_pred))
		Db = (-2 / n) * sum(Y - Y_pred)
		Dm = max(min(Dm, clip_value), -clip_value)
		Db = max(min(Db, clip_value), -clip_value)
		m = m - L * Dm
		b = b - L * Db
		cost = mean_square_error(X, Y, n, m, b)

        # If the cost is increasing, reduce the learning rate
		# print(f"========\nprev:{prev_cost}")
		# print(f"cost:{cost}")
		if not 0 < cost < prev_cost:
			L *= 0.9
		prev_cost = cost
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

	# calcul cost before training (diff between prediction and real value)
	cost = mean_square_error(X, Y, n, m, b)
	print(f"cost before training: {cost}")

	# gradient descent
	
	m, b = gradient_descent(X, Y, n, m, b)
	# iterate

	# calcul cost after training (diff between prediction and real value)
	cost = mean_square_error(X, Y, n, m, b)
	print(f"cost after training: {cost}")
	
	# make predictons
	# print(f"prediction: {m * 7000 + b}")

	# # evaluate

	# # visualize
	plot_predictions(X, Y, get_Yprediction(m, X, b))