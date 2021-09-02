import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zlib

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(args):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    batch_size = 32
    data_path = 'data'
    model_path = os.path.join("models", "ex1", args.version)
    model_tflite_name = f"Group18_th_{args.version}.tflite"
    tflite_model_path = os.path.join("models", "ex1", "tflite_models", args.version, model_tflite_name)

    if os.path.exists(os.path.dirname(tflite_model_path)) is False:
        os.makedirs(os.path.dirname(tflite_model_path))

    train_dataset, val_dataset, test_dataset = fetch_datasets(data_path, batch_size)

    # options & params
    model_type = 'mlp'  # cnn or mlp
    width_scaling = 0.1
    pruning_final_sparsity = 0.85
    epochs = 22

    model = build_model(model_type, width_scaling=width_scaling)

    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                               final_sparsity=pruning_final_sparsity,
                                                                               begin_step=len(train_dataset) * 5,
                                                                               end_step=len(train_dataset) * 15)}
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model = prune_low_magnitude(model, **pruning_params)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    input_shape = [batch_size, 6, 2]
    model.build(input_shape)
    model.compile(optimizer='adam', loss='mse', metrics=[MultiOutputMAE()])
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
    test_performance = model.evaluate(test_dataset, return_dict=True)
    model = tfmot.sparsity.keras.strip_pruning(model)
    print("Full model performance: ", test_performance)
    model.save(model_path)
    # convert to tflite
    converter_optimisations = [tf.lite.Optimize.DEFAULT]
    tflite_size = to_tflite(model_path, tflite_model_path, converter_optimisations=converter_optimisations, compressed=True)
    print(f"tflite size: {round(tflite_size, 3)} KB", )
    # test tflite model
    tflite_performance = test_tflite(tflite_model_path, test_dataset)
    print("tflite performance: ", tflite_performance)





def to_tflite(source_model_path, tflite_model_path, converter_optimisations=None, compressed=False):

    converter = tf.lite.TFLiteConverter.from_saved_model(source_model_path)

    if converter_optimisations is not None:
        converter.optimizations = converter_optimisations
    tflite_model = converter.convert()

    if not os.path.exists(os.path.dirname(tflite_model_path)):
        os.makedirs(os.path.dirname(tflite_model_path))

    with open(tflite_model_path, 'wb') as fp:
        fp.write(tflite_model)

    if compressed:
        compressed_tflite_model_path = tflite_model_path+".zlib"
        with open(compressed_tflite_model_path, 'wb') as fp:
            compressed_tflite_model = zlib.compress(tflite_model, level=9)
            fp.write(compressed_tflite_model)
        return os.path.getsize(compressed_tflite_model_path) / 1024

    return os.path.getsize(tflite_model_path) / 1024


def test_tflite(tflite_model_path, test_dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total, count = np.zeros(2), 0
    test_dataset = test_dataset.unbatch().batch(1)
    for features, labels in test_dataset:
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction = prediction.squeeze()
        labels = labels.numpy().squeeze()
        error = np.abs(prediction - labels)
        error = np.mean(error, 0)
        total += error
        count += 1

    return total/float(count)


def fetch_datasets(data_path, batch_size=32):
    csv_path = os.path.join(data_path, "jena_climate_2009_2016.csv")

    if not os.path.exists(csv_path):
        tf.keras.utils.get_file(origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
                                fname='jena_climate_2009_2016.csv.zip', extract=True, cache_dir='.', cache_subdir='data')

    jena_dataframe = pd.read_csv(csv_path)
    column_indices = [2, 5]  # temperature and humidity data columns
    columns = jena_dataframe.columns[column_indices]
    data = jena_dataframe[columns].values.astype(np.float32)

    # splitting dataframe in train/validation/test
    n = len(data)
    train_data = data[0:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    input_width, output_width = 6, 6

    # build datasets
    generator = WindowGenerator(batch_size, input_width, output_width, mean, std)
    train_dataset = generator.make_dataset(train_data, True)
    val_dataset = generator.make_dataset(val_data, False)
    test_dataset = generator.make_dataset(test_data, False)

    return train_dataset, val_dataset, test_dataset


def build_model(model='mlp', *args, **kwargs):
    if model == 'mlp':
        return make_mlp(*args, **kwargs)
    else:
        return make_cnn(*args, **kwargs)


def make_mlp(input_shape=(6, 2), units=128, output_units=12, width_scaling=1):
    model = keras.Sequential([keras.layers.Flatten(input_shape=input_shape),
                              keras.layers.Dense(units=units * width_scaling, activation='relu'),
                              keras.layers.Dense(units=units * width_scaling, activation='relu'),
                              keras.layers.Dense(units=output_units),
                              keras.layers.Reshape([6, 2])])

    return model


def make_cnn(input_shape=(6, 2), filters=64, units=64, output_units=12, width_scaling=1):
    model = keras.Sequential([keras.layers.Conv1D(filters=filters * width_scaling, kernel_size=3, input_shape=input_shape),
                              keras.layers.ReLU(),
                              keras.layers.Flatten(),
                              keras.layers.Dense(units=units * width_scaling, activation='relu'),
                              keras.layers.Dense(units=output_units),
                              keras.layers.Reshape([6, 2])])

    return model


class WindowGenerator:
    def __init__(self, batch_size, input_width, output_width, mean, std):
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :self.input_width, :]
        labels = features[:, self.input_width:, :]
        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_width, 2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train=True):
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.input_width + self.output_width,
                                                                       sequence_stride=1, batch_size=self.batch_size)
        dataset = dataset.map(self.preprocess)
        dataset = dataset.cache()
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='multi_output_mae', **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=[2])
        self.count = self.add_weight('count', initializer='zeros')

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1)
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str, choices=['a', 'b'], required=True, help='Model version to build')
    args = parser.parse_args()
    main(args)

