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

def augument_images(train_images, train_labels):
	x_ret = train_images
	y_ret = train_labels
	datagen = ImageDataGenerator(
			rotation_range=40,
			width_shift_range=0.2,
			height_shift_range=0.2,
			rescale=1.,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='nearest')
	
	# Create 5 batches of 60000 images
	it = datagen.flow(x=train_images, y=train_labels, batch_size=train_images.shape[0], shuffle=False)
	
	
	i = 0
	for x,y in it:
		x_ret = np.vstack((x_ret,x))
		y_ret = np.concatenate((y_ret, y))
		print(x.shape)
		print(x_ret.shape)
		print(y.shape)
		print(y_ret.shape)
		#print(i)
		#plot_image2(x[0])
			
		i = i + 1
		if i == 3:
			break
	return (x_ret, y_ret)

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
def nn_convnet(train_images, train_labels, test_images, test_labels, l2_reg=False):
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
	if l2_reg == True:
		model.add(keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
		model.add(keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
	else:
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
	model.fit(train_images, train_labels, epochs=10)
	
	# Evaluate model using test data
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

	print('\nTest accuracy:', test_acc)
	return test_acc
	

clothing_mnist = fashion_mnist

# train model with train_images and train_labels 
# Test the model accuracy with test_images and test_labels
(train_images, train_labels), (test_images, test_labels) = clothing_mnist.load_data()

class_names_clothing = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
						'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

# Add a new channel
train_images_re = train_images.reshape(train_images.shape[0], 28, 28, 1)

(train_images_aug,train_labels_aug) = augument_images(train_images_re, train_labels)
train_images_aug_corr = train_images_aug[:,:,:,0]
#plot_first_image()

#plot_first_25_images_with_class(class_names_clothing, train_images, train_labels)


test_acc_dict = {}
test_acc_dict["Fully connected, img aug"] = nn_fully_connected(train_images_aug_corr, train_labels_aug, test_images, test_labels)
test_acc_dict["Fully connected"] = nn_fully_connected(train_images, train_labels, test_images, test_labels)

test_acc_dict["Padding, small window, small strides, droput"] =   nn_convnet(train_images, train_labels, 
															test_images, test_labels, False)
test_acc_dict["Padding, small window, small strides, droput, img aug"] =   nn_convnet(train_images_aug_corr, train_labels_aug, 
															test_images, test_labels, False)
test_acc_dict["Padding, small window, small strides, dropout, l2 regularizer, img_aug"] =   nn_convnet(train_images_aug_corr, train_labels_aug, 
															test_images, test_labels, True)

print(test_acc_dict)
#predictions = model.predict(test_images)

#print(class_names_clothing[np.argmax(predictions[0])])
#plot_image(test_images, 0)









