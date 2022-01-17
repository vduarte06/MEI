import numpy as np
import matplotlib as plt

def size(X, dim=None):
	if dim:
		return np.shape(X[:,dim])[0]
	return X.shape

def ceil(x):
	return np.ceil(x)

def floor(x):
	return np.floor(x)

def sqrt(x):
	return np.sqrt(x)

def ones(x,y):
	return np.ones((int(x), int(y)))

def round(x):
	return np.round(x)

def max(x):
	return np.max(x)

def abs(x):
	return np.abs(x)

def displayData(X, example_width=None):
	#DISPLAYDATA Display 2D data in a nice grid
	#   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
	#   stored in X in a nice grid. It returns the figure handle h and the 
	#   displayed array if requested.

	# Set example_width automatically if not passed in
	if not example_width: 
		example_width = round(np.sqrt(size(X, 2)))
	# Gray Image
	#colormap(gray)

	# Compute rows, cols
	m, n = size(X)
	example_height = n / example_width

	# Compute number of items to display
	display_rows = int(floor(sqrt(m)))
	display_cols = int(ceil(m / display_rows))

	# Between images padding
	pad = 1

	# Setup blank display
	display_array =  ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad))
	print(display_array.shape)
	input()
	# Copy each example into a patch on the display array
	curr_ex = 1
	for j in range(1, display_rows):
		for i in range(1, display_cols):
			if curr_ex > m: 
				break 
			
			# Copy the patch
			
			# Get the max value of the patch
			max_val = max(abs(X[curr_ex, :]))
			x = (pad + (j - 1) * (example_height + pad) + np.arange(1,example_height))
			y = pad + (i - 1) * (example_width + pad) + np.arange(1, example_width)
			print(x.shape,y.shape)
			#display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ) = reshape(X(curur_ex, :), example_height, example_width) / max_val
			curr_ex = curr_ex + 1
			
		if curr_ex > m: 
			return 
		

	# Display Image
	#h = imagesc(display_array, [-1 1])

	# Do not show axis
	#axis image off

	#drawnow


