import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import HDF5Matrix
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.applications.densenet import DenseNet169
from IPython.display import clear_output


def make_data(x_dir, y_dir):
    x = np.array(HDF5Matrix(x_dir, 'x'))
    y = np.array(HDF5Matrix(y_dir, 'y')).reshape([-1, 1])
    return x, y

x_train_dir = 'camelyonpatch_level_2_split_train_x.h5'
y_train_dir = 'camelyonpatch_level_2_split_train_y.h5'
x_test_dir = 'camelyonpatch_level_2_split_test_x.h5'
y_test_dir = 'camelyonpatch_level_2_split_test_y.h5'
x_val_dir = 'camelyonpatch_level_2_split_valid_x.h5'
y_val_dir = 'camelyonpatch_level_2_split_valid_y.h5'

x_train, y_train = make_data(x_train_dir, y_train_dir)
x_test, y_test = make_data(x_test_dir, y_test_dir)
x_val, y_val = make_data(x_val_dir, y_val_dir)

batch_size = 128

train_datagen = ImageDataGenerator(rescale=1./255,
                                  vertical_flip = True,
                                  horizontal_flip = True,
                                  rotation_range=90,
                                  zoom_range=0.2,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.05,
                                  channel_shift_range=0.1)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

dropout_fc = 0.5

conv_base = DenseNet169(weights = 'imagenet', include_top = False, input_shape = (96,96,3))
#conv_base.summary()

my_model = Sequential()

my_model.add(conv_base)
my_model.add(Flatten())
my_model.add(Dense(256, use_bias=False))
my_model.add(BatchNormalization())
my_model.add(Activation("relu"))
my_model.add(Dropout(dropout_fc))
my_model.add(Dense(1, activation = "sigmoid"))

#print(my_model.summary())


#As we're using DenseNet169 trained on ImageNet, we're going to need to train the last few layers
#instead of the just the last one.
#Cell images are quite different to what you see on ImageNet.

conv_base.Trainable=True

'''set_trainable=False
for layer in conv_base.layers:
    if 'block30' in layer.name:
        print(layer.name)
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False'''

# training

from keras import optimizers
my_model.compile(optimizers.Adam(0.001), loss = "binary_crossentropy", metrics = ["accuracy"])

train_step_size = train_generator.n // train_generator.batch_size
valid_step_size = val_generator.n // val_generator.batch_size

earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=2, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)
check = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


history = my_model.fit_generator(train_generator,
                                     steps_per_epoch = train_step_size,
                                     epochs = 5,
                                     validation_data = val_generator,
                                     validation_steps = valid_step_size,
                                     callbacks = [reduce, earlystopper, check],
                                     verbose = 1)

my_model.save("dense.h5")
print("Saved model to disk")

# PLOTTING

epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.plot(epochs, history.history['acc'], color='blue', label="training_accuracy")
plt.plot(epochs, history.history['val_acc'], color='red',label="validation_accuracy")
plt.legend(loc='best')
plt.title('validation')
plt.xlabel('epoch')
plt.savefig("validation.png", bbox_inches='tight')
plt.show()


plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")
plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig("training.png", bbox_inches='tight')
plt.show()


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)

tta_steps = 5
predictionsTTA = []

for i in range(tta_steps):
    preds = my_model.evaluate_generator(test_generator, steps =1)
    predictionsTTA.append(preds[1])

print("Accuracy Score")
print(max(predictionsTTA))
