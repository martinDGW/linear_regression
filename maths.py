def get_x_sum(x_data):
	sum = 0
	for i in range(len(x_data)):
		sum = sum + x_data[i]
	return sum

def get_x_average(x_data):
	average = get_x_sum(x_data) / len(x_data)
	return average

def get_y_sum(y_data):
	sum = 0
	for i in range(len(y_data)):
		sum = sum + y_data[i]
	return sum

def get_y_average(y_data):
	average = get_y_sum(y_data) / len(y_data)
	return average

def get_xy_sum(x_data, y_data):
	sum = 0
	for i in range(len(x_data)):
		sum = sum + y_data[i] * x_data[i]
	return sum

def get_xy_average(x_data, y_data):
	average = get_xy_sum(x_data, y_data) / len(x_data)
	return average

def get_x2_sum(x_data):
	sum = 0
	for i in range(len(x_data)):
		sum = sum + x_data[i] * x_data[i]
	return sum

def get_x2_average(x_data):
	average = get_x2_sum(x_data) / len(x_data)
	return average

def get_y2_sum(y_data):
	sum = 0
	for i in range(len(y_data)):
		sum = sum + y_data[i] * y_data[i]
	return sum

def get_y2_average(y_data):
	average = get_y2_sum(y_data) / len(y_data)
	return average

def get_slope(x_data, y_data):
	return get_xy_average(x_data, y_data) - (get_x_average(x_data) * get_y_average(y_data)) / get_x2_average(x_data) - (get_x_average(x_data) * get_x_average(x_data))

def get_slope_and_b(x_data, y_data):
	slope = get_slope(x_data, y_data)
	b = get_y_average(y_data) - slope * get_x_average(x_data)
	return slope, b