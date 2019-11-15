import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np

# Plot one of the images
def plot_image(images, index):
	plt.figure()
	plt.imshow(images[index])
	plt.colorbar()
	plt.grid(False)
	plt.show()

def plot_image2(x):
	x_corr = x[:,:,0]
	plt.figure()
	plt.imshow(x_corr)
	plt.colorbar()
	plt.grid(False)
	plt.show()

def plot_image3(x):
	plt.figure()
	plt.imshow(x)
	plt.colorbar()
	plt.grid(False)
	plt.show()

# Plot 25 first images
def plot_first_25_images_with_class(class_names_clothing,
									train_images, train_labels):
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(train_images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names_clothing[train_labels[i]])
	plt.show()

def plot_first_32_images(model_layers_predictions):
	plt.figure(figsize=(10,10))
	for i in range(32):
		plt.subplot(4,8, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(model_layers_predictions[0, :, :, i], cmap=plt.cm.binary)
	plt.show()

# A fully connected neural network
def nn_fully_connected(train_images, train_labels, test_images, test_labels):
	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs=10)

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)
	return test_acc

# ---- References: 
# (1) DEEP LEARNING with Python, Francois Chollet
# (2) https://keras.io/layers/convolutional/
# (3) https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# (4) https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/
# (5) https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
# (6) https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
#
#
# ---- Method
# The CNN below was inspired by (6), where a CNN with first a layer 
# Conv2D and then a layer MaxPooling2D, followed by two more pairs
# of layers of the same type with increasing filter size. The CNN
# is then flatten with a Flatten layer, followed by two 128-unit Dense layer
# followed by a 10-unit Dense layer outputting the probabilities.
# Different settings was the tested and the result was printed.
# Dropout was added to prevent overfitting, and a option to use L2-
# regularization was added.
#
#
def nn_convnet(train_images, train_labels, test_images, test_labels, epochs_in=1):
	# Reshape to insert depth of 1
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28,1)

	
	model = keras.models.Sequential()
	window = (3,3) # End up with 26x26 layer output
	padding_in = 'same'
	strides_in = (1,1)

	
	model.add(keras.layers.Conv2D(32, window, activation='relu',
									input_shape=(28,28,1),
									data_format='channels_last',
									strides = strides_in,
									padding=padding_in))
	# Down-sample the layer output by applying MaxPooling2D
	# MaxPooling2D was chosen over AveragePooling2D because of better accuracy
	model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
	model.add(keras.layers.Dropout(0.25))
	
	model.add(keras.layers.Conv2D(64, window, activation='relu',
										padding=padding_in))
	# Down-sample the layer output by applying MaxPooling2D
	# MaxPooling2D was chosen over AveragePooling2D because of better accuracy
	model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
	model.add(keras.layers.Dropout(0.25))
	
	# With strides layer output is allready at 1x1
	model.add(keras.layers.Conv2D(128, (3,3), activation='relu',
										padding=padding_in))
	# Down-sample the layer output by applying MaxPooling2D
	# MaxPooling2D was chosen over AveragePooling2D because of better accuracy
	model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
	model.add(keras.layers.Dropout(0.25))
		
	# Flatten layer output and run it through a fully connected network
	model.add(keras.layers.Flatten())
	# Use L2-reularization

	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dropout(0.5))
	
	# According to (1) p114 softmax and categorial_crossentropy should be
	# used for multiclass single-label classification
	# 10 units corresponding to the 10 clothing classes, with
	# softmax activation to return probabilities summing to 1.
	model.add(keras.layers.Dense(10, activation='softmax'))
	
	model.summary()
	
	# Targets are integers so using sparse_categorical_crossentropy
	# Optimizer seemed the most used in references (3)
	model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	# Epoch 5 was used because overfitting was noticed after 5 epochs
	model.fit(train_images, train_labels, epochs=epochs_in)
	
	# Evaluate model using test data
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)
	return (test_acc, model)
	

clothing_mnist = fashion_mnist

# train model with train_images and train_labels 
# Test the model accuracy with test_images and test_labels
(train_images, train_labels), (test_images, test_labels) = clothing_mnist.load_data()

class_names_clothing = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
						'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Normalize image data between 0.0 and 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

(acc, model) = nn_convnet(train_images, train_labels, test_images, test_labels)

print(acc)

# Extract the output layers from the model
model_layer_outputs = [layer.output for layer in model.layers]          # all layer outputs
act_model = keras.models.Model(inputs=model.input, outputs=model_layer_outputs)

# Output for first image in test set
img_input = test_images[0]
# Reshape to expected dimensions (number_of_images, 28, 28, 1)
img_input = img_input.reshape(1, 28, 28, 1)
model_layers_prediction = act_model.predict(img_input)

# Print the number of layers with predictions
print(len(model_layers_prediction))

#print(model_layers_prediction)

# Specify index to plot images for (0-7)
layer_act = model_layers_prediction[7]
print(layer_act.shape)

plot_first_32_images(layer_act)
#plot_image3(first_layer_act[0, :, :, 3])






