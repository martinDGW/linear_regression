#!/usr/bin/env python3

import numpy as np
import sys
import signal
import configparser

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

FILE = "./DataSet/thetas"

def try_preditcion(a, b, x_multiplier, y_multiplier):
	c = ""
	val = ""
	while not c in ["n"]:
		c = input(f"{YELLOW}Do you want to try a prediction ? [Y/n]: {RESET}")
		if c in ["Y", "y"]:
			while not val.isdigit():
				val = input(f"How many kilometers does your vehicle have ? ")
			prediction = (a * (int(val) / x_multiplier) + b) * y_multiplier # denormalized value
			val = ""
			if prediction < 0:
				print(f"{RED}Model has predicted a negative value.\nYou should not sell your vehicle...{RESET}")
			else:
				print(f"Estimated price: {int(prediction)}")

def signal_handler(sig, frame):
	print(f"{RED}Signal detected.{RESET}")
	sys.exit(1)

def get_parameters(file_name):

	try:
		config = configparser.ConfigParser()
		config.read(file_name)

		a = float(config["Parameters"]["a"])
		b = float(config["Parameters"]["b"])
		x_multiplier = float(config['Parameters']['x_multiplier'])
		y_multiplier = float(config['Parameters']['y_multiplier'])

		return a, b, x_multiplier, y_multiplier
	except Exception as e:
		print(f"{RED}Model does not seem to be trained.{RESET}")
		return 0, 0, 1, 1


if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)

	theta0, theta1, x_multiplier, y_multiplier = get_parameters(FILE)
		
	try_preditcion(theta0, theta1, x_multiplier, y_multiplier)
