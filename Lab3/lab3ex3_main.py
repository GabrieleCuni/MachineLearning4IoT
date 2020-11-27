# Standard library imports
from time import time
import sys
import argparse
import os

# Third party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Local application imports
from mymodules.preprocessing import WindowGenerator
from mymodules.metrics import MultiOutputMAE


def load_data():
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True,
        cache_dir='.', 
        cache_subdir='data'
    )
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    column_indices = [2, 5]
    columns = df.columns[column_indices]
    data = df[columns].values.astype(np.float32)
    n = len(data)
    train_data = data[0:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]

    return train_data, val_data, test_data

def initialize_model(model_name, label_options, filepath):
    if label_options == 0 or label_options == 1:
        units = 1
    elif label_options == 2:
        units = 2
    else:
        print("Error: -l, --labels [0 1 2]")
        sys.exit(1)

    if model_name == "MLP":
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=units)
        ])
        filepath = os.path.join(filepath, "MLP")
    elif model_name == "CNN":
        model = keras.Sequential([
            keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=units)
        ])
        filepath = os.path.join(filepath, "CNN")
    elif model_name == "LSTM":
        model = keras.Sequential([
            keras.layers.LSTM(units=64),
            keras.layers.Flatten(),
            keras.layers.Dense(units=units)
        ])
        filepath = os.path.join(filepath, "LSTM")
    else:
        print("Error: -m, --model [MLP, CNN, LSTM]")
        sys.exit(1)

    if label_options == 0:
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()] # tf.keras.metrics.MeanAbsoluteError()
        )
        filepath = os.path.join(filepath, "0")
    elif label_options == 1:
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()] # tf.keras.metrics.MeanAbsoluteError()
        )
        filepath = os.path.join(filepath, "1")
    elif label_options == 2:
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[MultiOutputMAE()] 
        )
        filepath = os.path.join(filepath, "2")
    else:
        print("Error: -l, --labels [0 1 2]")
        sys.exit(1)

    return model, filepath


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', type=str, default="MLP", help='model name')
    parser.add_argument("-l", '--labels', type=int, default=0, help='model output')
    parser.add_argument("-o", "--output", type=str, default="./modelsEx3", help="Model output directory. Default: ./models")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs. Default 20")
    parser.add_argument("--input-width", type=int, default=6, help="Time series window width. Defaut 6")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args(argv)

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    train_data, val_data, test_data = load_data()
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    if args.verbose:
        print(f"Mean shape: {mean.shape}") # Mean shape: (2,)
        print(f"Std shape: {std.shape}") # Std shape: (2,)

    generator = WindowGenerator(args.input_width, args.labels, mean, std, verbose=args.verbose)
    train_dataset = generator.make_dataset(train_data, True)
    val_dataset = generator.make_dataset(val_data, False)
    test_dataset = generator.make_dataset(test_data, False)

    model, filepath = initialize_model(args.model, args.labels, args.output)

    if args.verbose:
        model.summary()

    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)         

    if os.path.exists(filepath) is False:
        os.makedirs(filepath)
    model.save(filepath) # save anything

    # model = keras.models.load_model('path/to/location’)

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test. Loss {test_loss} Accuracy {test_acc}")
    with open( os.path.join(filepath, f"Test_loss: {test_loss} - Test accuracy: {test_acc}.log"), "w" ) as f:
        f.write("")


if __name__=="__main__":
    main(sys.argv[1:])