import numpy as np
import pandas as pd
import os

# Use early stopping to terminate training epochs through callbacks
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import network libraries
import keras
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D

from sklearn.metrics import accuracy_score

## Define class for Task B1
class CNN_B1:
	# Callback function to interrupt the overfitting model
	def callback_func(self, B1_dir):
		# Seek a mininum for validation loss and display the stopped epochs using verbose and adding delays
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

		# Save best model using checkpoint
		model_path = os.path.join(B1_dir, 'VGGNet.h5')
		mcp = ModelCheckpoint(os.path.normcase(model_path), monitor='val_loss', mode='min', verbose=1, save_best_only=True)

		# Define callback function in a list
		callback_list = [es, mcp]

		return callback_list, model_path

	# Training CNN network and save the model
	def train(self, myDir, num_class, train_generator, valid_generator, eval_generator):
		cb_list, CNN_model_path = self.callback_func(myDir)

		# Create a sequential model
		model = Sequential()

		# Add 1st convolution block
		model.add(Conv2D(filters=16, input_shape=train_generator.image_shape, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

		# Add 2nd convolution block
		model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

		# Add 3rd convolution block
		model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"))

		# Add 4th convolution block
		model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

		# Output Layer (5 clases for face shapes)
		model.add(Flatten())
		model.add(Dense(units=64,activation="relu"))
		model.add(Dropout(0.5))
		model.add(Dense(units=num_class, activation="softmax"))

		# Display summary of the model
		model.summary()

		# Compile the model using ADAM (Adaptive learning rate optimization)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		# Set steps per epoch for callback 
		STEP_SIZE_TRAIN = train_generator.samples//train_generator.batch_size
		STEP_SIZE_VALID = valid_generator.samples//valid_generator.batch_size

		history = model.fit_generator(generator=train_generator,
		                              steps_per_epoch=STEP_SIZE_TRAIN,
		                              epochs=30,
		                              callbacks=cb_list,
		                              validation_data=valid_generator,
		                              validation_steps=STEP_SIZE_VALID)

		eval_model = model.evaluate_generator(generator=eval_generator, steps=STEP_SIZE_VALID, verbose=1)
		print('Training '+ str(model.metrics_names[1]) + ': '  + str(eval_model[1]))
		train_acc = {'CNN-softmax': eval_model[1]}

		return train_acc, CNN_model_path


	def test(self, model_path, test_generator):
		# Fit the model to the test dataset by loading the model 
		saved_model = load_model(model_path)

		# Predict the face shape
		STEP_SIZE_TEST = test_generator.samples//test_generator.batch_size
		test_generator.reset()
		pred = saved_model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)

		# Determine the maximum activation value at the output layers for each sample
		pred_class = np.argmax(pred, axis=1)   # axis = 1 give max value along each row

		# True labels of test dataset
		true_class = test_generator.classes

		# Accuracy score
		test_score = accuracy_score(true_class, pred_class)

		test_acc = {'CNN-softmax': test_score}

		return test_acc