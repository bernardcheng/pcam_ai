import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import Callback
import keras.backend as K
from keras.utils import HDF5Matrix
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications import ResNet50
# from keras.applications.densenet import DenseNet121
# from keras.applications.nasnet import NASNetMobile
# from keras.applications.densenet import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras_OneCycle import OneCycle

def make_data(x_dir, y_dir, normalise = False):
    x = np.array(HDF5Matrix(x_dir, 'x'))
    y = np.array(HDF5Matrix(y_dir, 'y'))
    y = np.reshape(y, y.shape[0])

    if normalise == True:
        x_min = x.min(axis=(1, 2), keepdims=True)
        x_max = x.max(axis=(1, 2), keepdims=True)

        x = (x - x_min)/(x_max-x_min)

    return x, y


def make_model(base_model, OPTIMIZER, LOSS):
    print("Making Model...")
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # predictions = Dense(1,activation='softmax')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    for layer in base_model.layers[:-1]:
        layer.trainable=False

    # for layer in model.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics = ['accuracy'])
    print("Made Model!")
    return model


def generate_batches(x_train, y_train, BATCH_SIZE, shuffle = False):

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                channel_shift_range=10,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest')
    data_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=shuffle)
    return data_generator

def train_model(train_generator, val_generator, test_generator, x_train, x_val):
    print('Starting Training...')
    # sched = OneCycle(min_lr=7e-3, max_lr=7e-2, min_mtm = 0.85, max_mtm = 0.95, annealing_stage=0.1, annealing_rate=0.01,
    #       training_iterations=np.ceil(((x_train.shape[0]*EPOCHS)/(BATCH_SIZE))))
    # fit_history = model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=len(x_train) // BATCH_SIZE,
    #         epochs=EPOCHS,
    #         validation_data = val_generator,
    #         validation_steps= len(x_val) // BATCH_SIZE,
    #         callbacks = [sched]
    #         )
    fit_history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(x_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data = val_generator,
            validation_steps= len(x_val) // BATCH_SIZE,
            )
    print("Finished Training!")

    plt.figure(1, figsize = (15,8))

    plt.subplot(221)
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])

    plt.subplot(222)
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])

    plt.show()
    plt.savefig("Result_plot.PNG")

    model.save("saved_models/Resnet50_model.h5")
    print("Saved model to disk.")
    test_generator.reset()
    pred_score = model.evaluate_generator(test_generator, steps = len(test_generator))

    return pred_score


x_train_dir = 'camelyonpatch_level_2_split_train_x.h5'
y_train_dir = 'camelyonpatch_level_2_split_train_y.h5'
x_test_dir = 'camelyonpatch_level_2_split_test_x.h5'
y_test_dir = 'camelyonpatch_level_2_split_test_y.h5'
x_val_dir = 'camelyonpatch_level_2_split_valid_x.h5'
y_val_dir = 'camelyonpatch_level_2_split_valid_y.h5'

img_width, img_height = 96, 96 # For Resnet: (96,96), For NasNetMobile: (224,224)

BATCH_SIZE = 32
EPOCHS = 2
FREEZE_LAYERS = 150  # freeze the first this many layers for training

base_model = ResNet50(weights='imagenet',include_top=False, input_shape=(img_width, img_height, 3))


OPTIMIZER = SGD(lr=0.01) #Adam(1e-4)
LOSS = 'binary_crossentropy'

if __name__ == "__main__":
    x_train, y_train = make_data(x_train_dir, y_train_dir)
    x_test, y_test = make_data(x_test_dir, y_test_dir)
    x_val, y_val = make_data(x_val_dir, y_val_dir)

    print("Generated Data!")

    model = make_model(base_model, OPTIMIZER, LOSS)

    train_generator = generate_batches(x_train, y_train, BATCH_SIZE, shuffle = True)
    val_generator = generate_batches(x_val, y_val, BATCH_SIZE)
    test_generator = generate_batches(x_test, y_test, 1)
    print("Generated Batches!")

    accuracy_score = train_model(train_generator, val_generator, test_generator, x_train, x_val)
    # model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=len(x_val) // BATCH_SIZE,
    #         EPOCHS=EPOCHS
    #         )
    pred_score = model.evaluate_generator(test_generator, steps = len(test_generator))
    print(pred_score)
    print("Test accuracy: ", accuracy_score)

    #new_model = load_model('resnet_model.h5')
    #new_model.summary()

    #return accuracy_score


    #run()
