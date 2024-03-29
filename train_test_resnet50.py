import os, keras, argparse
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Input, Dropout, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
from keras.utils import HDF5Matrix, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from keras_pcam.dataset.pcam import load_data

def limit_gpu():
    """Limit usage of GPU to avoid cuda out of memory."""
    # TensorFlow wizardry
    config = tf.ConfigProto()
    
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    # Create a session with the above options specified.
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def load_data(data_path='/home/ubuntu/data/patchcamelyon'):
    """Load PCam dataset"""
    print("# Loading dataset from PCam...")
    train_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_train_x.h5')
    train_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_train_y.h5')
    val_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_valid_x.h5')
    val_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_valid_y.h5')
    test_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_test_x.h5')
    test_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_test_y.h5')

    x_train = np.array(HDF5Matrix(train_x_path, 'x', start=0, end=train_size))
    y_train = np.array(HDF5Matrix(train_y_path, 'y', start=0, end=train_size)).reshape([-1,1])
    x_val = np.array(HDF5Matrix(val_x_path, 'x', start=0, end=val_size))
    y_val = np.array(HDF5Matrix(val_y_path, 'y', start=0, end=val_size)).reshape([-1,1])
    x_test = np.array(HDF5Matrix(test_x_path, 'x', start=0, end=val_size))
    y_test = np.array(HDF5Matrix(test_y_path, 'y', start=0, end=val_size)).reshape([-1,1])
    print("# Loaded train data {0}, val data {1}, test data {2}.".format(x_train.shape, x_val.shape, x_test.shape))
    return x_train, y_train, x_val, y_val, x_test, y_test

def train(x_train, y_train, x_val, y_val):
    """Train a model"""
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
    base_model = ResNet50(include_top=False, input_tensor=new_input)

    # Let's take a look to see how many layers are in the base model (175 Resnet50)
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine tune from this layer onwards
    fine_tune_at = 165

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    # add new classifier layers
    x = base_model(new_input)
    x = Dropout(0.5)(x)
    x = GlobalMaxPooling2D()(x)
    out = Dense(1, activation="sigmoid")(x)

    # define new model
    model = Model(inputs=new_input, outputs=out)
    # Specify the training configuration (optimizer, loss, metrics)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),  # Optimizer
                # Loss function to minimize
                loss=keras.losses.binary_crossentropy,
                # List of metrics to monitor
                metrics=['acc'])
    model.summary()
    # plot_model(model, to_file='model.png')

    print('# Fit model on training data')
    h5_path = os.path.join(output_dir,"model_resnet50_{}_epochs_{}_lr_{}_idx.h5".format(epochs,learning_rate,idx))
    checkpoint = ModelCheckpoint(h5_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data = val_datagen.flow(x_val, y_val, batch_size=batch_size),
                            epochs=epochs, verbose=1,
                            callbacks=[checkpoint],
                            steps_per_epoch=len(x_train) // batch_size, 
                            validation_steps=len(x_val) // batch_size)

    print('\nhistory dict:', history.history)
    return history

def plot(history):
    """Plot training & validation accuracy values"""
    plt.figure(1)
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(output_dir, "train_acc_resnet50_{}_epochs_{}_lr_{}_idx.png".format(epochs,learning_rate,idx)))

    # Plot training & validation loss values
    plt.figure(2)
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(output_dir, "train_loss_resnet50_{}_epochs_{}_lr_{}_idx.png".format(epochs,learning_rate,idx)))

def test(x_test, y_test):
    """Test the model with highest val accuracy."""
    #datagen for test
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # load model and specify a new input shape for images
    best_model_path = os.path.join(output_dir, 'model_resnet50_{}_epochs_{}_lr_{}_idx.h5'.format(epochs,learning_rate,idx))
    best_model = tf.keras.models.load_model(best_model_path)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = best_model.evaluate(test_datagen.flow(x_test, y_test, batch_size=batch_size))
    print('test loss, test acc:', results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, required=False, default=1, 
        help="Index number when saving graphs and model")
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10, 
        help="Number of epochs for train & val")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=1e-4, 
        help="Learning rate for optimizer")
    parser.add_argument('-b', '--batch', type=int, required=False, default=32, 
        help="Number of batch size")
    parser.add_argument('-d', '--data', type=str, required=False, default='/home/ubuntu/data/patchcamelyon', 
        help="Dataset path")
    parser.add_argument('-ts', '--train_size', type=int, required=False, default=2**18, 
        help="Number of train dataset")
    parser.add_argument('-vs', '--val_size', type=int, required=False, default=2**15, 
        help="Number of val & test dataset")
    parser.add_argument('-o', '--output', type=str, required=False, default='output', 
        help="Directory for output")
    parser.add_argument('-l', '--limit', type=bool, required=False, default=True, 
        help="Limit GPU usage to avoid out of memory")
    args = parser.parse_args()

    idx = args.idx
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch
    data_path = args.data
    train_size = args.train_size
    val_size = args.val_size
    output_dir = args.output
    limit = args.limit

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if limit:
        limit_gpu()
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_path)
    history = train(x_train, y_train, x_val, y_val)
    plot(history)
    test(x_test, y_test)