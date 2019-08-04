import os, keras, argparse
from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.models import Model, load_model
from keras.utils import HDF5Matrix, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def load_test_data():
    print("# Loading PCam test dataset...")
    test_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_test_x.h5')
    test_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_test_y.h5')

    x_test = np.array(HDF5Matrix(test_x_path, 'x', start=0, end=val_size))
    y_test = np.array(HDF5Matrix(test_y_path, 'y', start=0, end=val_size)).reshape([-1,1])
    print("# Loaded x_test {0} y_test {1}.".format(x_test.shape, y_test.shape))
    return x_test, y_test

def test(x_test, y_test):
    """Test best model."""
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # load model and specify a new input shape for images
    best_model_path = os.path.join(output_dir, best_model_file)
    best_model = tf.keras.models.load_model(best_model_path)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = best_model.evaluate(test_datagen.flow(x_test, y_test, batch_size=batch_size))
    print('test loss, test acc:', results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, required=False, default=32, 
        help="Number of batch size")
    parser.add_argument('-d', '--data', type=str, required=False, default='/home/ubuntu/data/patchcamelyon', 
        help="Dataset path")
    parser.add_argument('-m', '--model', type=str, required=False, default='model_10_epochs.h5', 
        help="Dataset path")
    parser.add_argument('-vs', '--val_size', type=int, required=False, default=2**15, 
        help="Number of train dataset")
    parser.add_argument('-o', '--output', type=str, required=False, default='output', 
        help="Directory for output")
    parser.add_argument('-l', '--limit', type=bool, required=False, default=True, 
        help="Limit GPU usage to avoid out of memory")
    args = parser.parse_args()

    batch_size = args.batch
    data_path = args.data
    best_model_file = args.model
    val_size = args.val_size
    output_dir = args.output
    limit = args.limit

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if limit:
        limit_gpu()
    x_test, y_test = load_test_data()
    test(x_test, y_test)