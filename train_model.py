#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import signal

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

FOLDER = "./DataSet"

## NEED PARSING
# === PARSING === #
def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument("--file", "-f", type=str, help='Path to the CSV file')

	return parser.parse_args()

# === GETTERS === #
def get_data_values(args):
	try:
		if args.file:
			csv_path = args.file
			data = pd.read_csv(csv_path)
			return data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy()
		else:
			raise ValueError(f"{RED}Please provide a path to the CSV file using the --file argument.{RESET}")
	except Exception as e:
		print(f"{e}")

def get_divider(nb):
	divider = 1
	while not 0 < nb / divider < 1:
		divider *= 10
	return divider

def normalize(array):
	larger_val = np.max(np.abs(array))
	if not 0 <= larger_val <= 1:
		divider = get_divider(larger_val)
		array = array.astype(float)
		array /= divider

	return array, divider

def get_thetas(a, b, x_divider, y_divider):
	try:
		with open(f"{FOLDER}/thetas", "w") as file:
			file.write("[Parameters]" + "\n")
			file.write("a = " + str(a) + "\n")
			file.write("b = " + str(b) + "\n")
			file.write("x_multiplier = " + str(x_divider) + "\n")
			file.write("y_multiplier = " + str(y_divider))
	except Exception as e:
		print(e)

# === VISUALIZE DATAS === #
def display_datas(X, Y, Y_pred, axs, epochs, costs):
	# display datas cloud
	axs[0].scatter(X, Y, label='value', color="blue", marker='o')
	axs[0].set_xlabel('Kms')
	axs[0].set_ylabel('Price')
	axs[0].set_title('data.csv representation')

	# display prediction line
	axs[0].plot(X, Y_pred, color= "red")

	# display cost evolution during training
	axs[1].set_xlabel('Iterations')
	axs[1].set_ylabel('Cost')
	axs[1].set_title('Cost evolution during iterations')
	axs[1].plot(np.arange(1, epochs + 1), costs, label='Cost Function')
	plt.show()

# === MATHS FUNCTIONS === #
def get_Ypredictions(a, X, b):
	return a * X + b

def cost_function(X, Y, n, a, b):
	## Mean Square Error
	# cost = (1 / (2 * n)) * (np.sum((Y - (a * X + b)) ** 2))
	cost = (1 / n) * (np.sum(np.square(Y - (a * X + b))))
	return cost

def gradient_descent(X, Y, n, a, b, L):
	Y_pred = get_Ypredictions(a, X, b)
	# calculer la derivee de notre courbe
	Da = (1 / n) * np.sum(X * (Y_pred - Y))
	Db = (1 / n) * np.sum(Y_pred - Y)
	# calculer les valeurs a(i + 1) et b(i + 1)
	# == les valeurs de a et b au step suivant
	a_next = a - L * Da
	b_next = b - L * Db
	return a_next, b_next

# === MAIN === #
def signal_handler(sig, frame):
	print(f"{RED}Signal detected.{RESET}")
	sys.exit(1)

def accuracy_percentage(Y, Y_pred):
	true_mean = np.mean(Y)
	numerator = np.sum(np.square(Y - Y_pred))
	denominator = np.sum(np.square(Y - true_mean))
	rse = numerator / denominator
	rrmse = np.sqrt(rse)
	print(f"RSE:{rse:.3f}")
	print(f"RRMSE:{rrmse:.3f}")

if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)

	args = parse_arguments()

	X, Y = get_data_values(args)
	X, x_divider = normalize(X)
	Y, y_divider = normalize(Y)
	# initialize parameters
	a		= 0				# slope
	b		= 0				# intercept
	n		= len(X)		# n values in dataset
	L		= 0.5			# learning rate
	epochs	= 4000			# number of iterations
	costs	= [0] * epochs	# cost evolution during iterations
	
	## iterative - gradient descent
	for i in range(epochs):
		costs[i] = cost_function(X, Y, n, a, b) * y_divider
		a, b = gradient_descent(X, Y, n, a, b, L)
	print(f"{GREEN}Your model is trained !{RESET}")

	## visualize
	fig, axs = plt.subplots(1, 2)
	get_thetas(a, b, x_divider, y_divider)
	accuracy_percentage(Y, get_Ypredictions(a, X, b))
	display_datas(X * x_divider, Y * y_divider, get_Ypredictions(a, X, b) * y_divider, axs, epochs, costs)
