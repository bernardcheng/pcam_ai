# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import os, keras
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras.utils import HDF5Matrix, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.nasnet import preprocess_input

# from keras_pcam.dataset.pcam import load_data

###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
###################################

SAVE_DIR = "output"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("# Loading whole dataset from PCam...")
dataset_path = '/home/ubuntu/data/pcam'
train_x_path = os.path.join(dataset_path, 'camelyonpatch_level_2_split_train_x.h5')
train_y_path = os.path.join(dataset_path, 'camelyonpatch_level_2_split_train_y.h5')
val_x_path = os.path.join(dataset_path, 'camelyonpatch_level_2_split_valid_x.h5')
val_y_path = os.path.join(dataset_path, 'camelyonpatch_level_2_split_valid_y.h5')
test_x_path = os.path.join(dataset_path, 'camelyonpatch_level_2_split_test_x.h5')
test_y_path = os.path.join(dataset_path, 'camelyonpatch_level_2_split_test_y.h5')

x_train = np.array(HDF5Matrix(train_x_path, 'x'))
y_train = np.array(HDF5Matrix(train_y_path, 'y'))
x_val = np.array(HDF5Matrix(val_x_path, 'x'))
y_val = np.array(HDF5Matrix(val_y_path, 'y'))
x_test = np.array(HDF5Matrix(test_x_path, 'x'))
y_test = np.array(HDF5Matrix(test_y_path, 'y'))

print("# Loaded train data {0}, val data {1}, test data {2}.".format(x_train.shape, x_val.shape, x_test.shape))

y_train = y_train.reshape([-1,1])
y_val = y_val.reshape([-1,1])
y_test = y_test.reshape([-1,1])

# datagen for training
datagen = ImageDataGenerator(
              preprocessing_function=preprocess_input,
              width_shift_range=4,  # randomly shift images horizontally
              height_shift_range=4,  # randomly shift images vertically 
              horizontal_flip=True,  # randomly flip images
              vertical_flip=True)  # randomly flip images

#datagen for validating
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# load model and specify a new input shape for images
new_input = Input(shape=(96, 96, 3))
base_model = NASNetMobile(include_top=False, input_tensor=new_input)

# mark loaded layers as not trainable
for layer in base_model.layers:
	layer.trainable = False

# add new classifier layers
x = base_model(new_input)
out1 = GlobalMaxPooling2D()(x)
out2 = GlobalAveragePooling2D()(x)
out3 = Flatten()(x)
out = Concatenate(axis=-1)([out1, out2, out3])
out = Dropout(0.5)(out)
out = Dense(1, activation="sigmoid", name="3_")(out)

# define new model
model = Model(inputs=new_input, outputs=out)
# Specify the training configuration (optimizer, loss, metrics)
model.compile(optimizer=keras.optimizers.Adam(0.0001),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.binary_crossentropy,
              # List of metrics to monitor
              metrics=['acc'])
model.summary()
# plot_model(model, to_file='model.png')

print('# Fit model on training data')
batch_size = 32
epochs = 10
h5_path = os.path.join(SAVE_DIR,"model_{}_epochs.h5".format(epochs))
checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data = val_datagen.flow(x_val, y_val, batch_size=batch_size),
                        epochs=epochs, verbose=1,
                        callbacks=[checkpoint],
                        steps_per_epoch=len(x_train) // batch_size, 
                        validation_steps=len(x_val) // batch_size)

print('\nhistory dict:', history.history)

# Plot training & validation accuracy values
plt.figure(1)
plt.clf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(os.path.join(SAVE_DIR, "train_acc_{}_epochs.png".format(epochs)))

# Plot training & validation loss values
plt.figure(2)
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(os.path.join(SAVE_DIR, "train_loss_{}_epochs.png".format(epochs)))

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=batch_size)
print('test loss, test acc:', results)