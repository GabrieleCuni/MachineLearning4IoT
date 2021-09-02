import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow import keras
import zlib
import tensorflow_model_optimization as tfmot
from time import time
from preprocessing import SignalGenerator

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main(model_version):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_path = os.path.join("data")
    tf_dataset_path = os.path.join(data_path, "tf_datasets")
    model_path = os.path.join("models", model_version)
    tflite_model_name = f"{model_version}.tflite"
    tflite_model_path = os.path.join("models", "tflite_models", tflite_model_name)

    if os.path.exists(os.path.dirname(tflite_model_path)) is False:
        os.makedirs(os.path.dirname(tflite_model_path))

    if os.path.exists(tf_dataset_path) is False:
        make_tf_datasets(data_path, tf_dataset_path)

    if model_version == "big":
        train_dataset, val_dataset, test_dataset = load_data_from_disk(tf_dataset_path, mfcc=True)
        epochs = 25

        model = make_big_model()
        model.compile(optimizer=tf.optimizers.Adam(0.01),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=keras.metrics.SparseCategoricalAccuracy())

        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        test_loss, test_acc = model.evaluate(test_dataset)
        model.save(model_path)
        to_tflite(model_path, tflite_model_path, compressed=False)

    else:  # little
        width_scaling = 0.29
        epochs = 30
        pruning_final_sparsity = 0.32

        train_dataset, val_dataset, test_dataset = load_data_from_disk(tf_dataset_path, mfcc=False)        
        model = make_little_model(width_scaling=width_scaling)
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                                   final_sparsity=pruning_final_sparsity,
                                                                                   begin_step=len(train_dataset) * 5,
                                                                                   end_step=len(train_dataset) * 27)}
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        model = prune_low_magnitude(model, **pruning_params)
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=keras.metrics.SparseCategoricalAccuracy())

        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
        test_loss, test_acc = model.evaluate(test_dataset)
        model = tfmot.sparsity.keras.strip_pruning(model)
        model.save(model_path)
        print(f"Full model: Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

        converter_optimisations = [tf.lite.Optimize.DEFAULT]  # PTQ
        tflite_model_size = to_tflite(model_path, tflite_model_path, converter_optimisations=converter_optimisations, compressed=True)
        print(f"tflite size: {round(tflite_model_size, 3)} KB", )

        tflite_performance = test_tflite(tflite_model_path, test_dataset)
        print("tflite performance: ", tflite_performance)

    return


def make_big_model():
    model = make_ds_cnn()
    return model

def make_little_model(width_scaling):
    # DSCNN STFT
    strides = [2, 2]
    input_shape = (32, 32, 1)
    units=8

    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * width_scaling), kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=int(256 * width_scaling), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=int(256 * width_scaling), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=units)
    ])

    return model




def make_ds_cnn(units=8, strides=(2, 1), input_shape=(49, 10, 1)):
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=input_shape, filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=512*2, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=units)
    ])

    return model


def to_tflite(source_model_path, tflite_model_path, converter_optimisations=None, representative_dataset=None,
              supported_ops=None, compressed=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(source_model_path)

    if converter_optimisations is not None:
        converter.optimizations = converter_optimisations

    if representative_dataset is not None:
        converter.representative_dataset = representative_dataset
    if supported_ops is not None:
        converter.target_spec.supported_ops = supported_ops
    tflite_model = converter.convert()

    if not os.path.exists(os.path.dirname(tflite_model_path)):
        os.makedirs(os.path.dirname(tflite_model_path))

    with open(tflite_model_path, 'wb') as fp:
        fp.write(tflite_model)

    if compressed:
        compressed_tflite_model_path = tflite_model_path + ".zlib"
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

    accuracy, count = 0, 0
    test_dataset = test_dataset.unbatch().batch(1)
    for features, labels in test_dataset:
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction = prediction.squeeze()
        prediction = np.argmax(prediction)
        labels = labels.numpy().squeeze()
        accuracy += prediction == labels
        count += 1

    return accuracy / float(count)


def make_tf_datasets(data_path, tf_dataset_path):
    dataset_path = os.path.join(data_path, "mini_speech_commands")
    if not os.path.exists(dataset_path):
        tf.keras.utils.get_file(origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                fname='mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir=data_path)
    with open("./kws_train_split.txt", "r") as fp:
        train_files = [line.rstrip() for line in fp.readlines()]  # len 6400
    with open("./kws_val_split.txt", "r") as fp:
        val_files = [line.rstrip() for line in fp.readlines()]  # len 800
    with open("./kws_test_split.txt", "r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  # len 800

    # labels = np.array(tf.io.gfile.listdir(str(dataset_path)))
    # labels = labels[labels != 'README.md']
    labels = ['down', 'stop', 'right', 'left', 'up', 'yes', 'no', 'go']

    STFT_OPTIONS = {'stft_frame_length': 256,
                    'stft_frame_step': 128,
                    'mfcc': False}

    MFCC_OPTIONS = {'stft_frame_length': 640,
                    'stft_frame_step': 320,
                    'mfcc': True,
                    'lower_freq_mel': 20,
                    'upper_freq_mel': 4000,
                    'num_mel_bins': 40,
                    'num_coefficients': 10}

    options = MFCC_OPTIONS
    tf_dataset_mfcc_path = os.path.join(tf_dataset_path, "mfcc")
    generator = SignalGenerator(labels, 16000, **options)
    train_dataset = generator.make_dataset(train_files, True)  # 200
    val_dataset = generator.make_dataset(val_files, False)  # 25
    test_dataset = generator.make_dataset(test_files, False)  # 25
    tf.data.experimental.save(train_dataset, os.path.join(tf_dataset_mfcc_path, 'th_train'))
    tf.data.experimental.save(val_dataset, os.path.join(tf_dataset_mfcc_path, 'th_val'))
    tf.data.experimental.save(test_dataset, os.path.join(tf_dataset_mfcc_path, 'th_test'))

    options = STFT_OPTIONS
    tf_dataset_stft_path = os.path.join(tf_dataset_path, "stft")
    generator = SignalGenerator(labels, 16000, **options)
    train_dataset = generator.make_dataset(train_files, True)  # 200
    val_dataset = generator.make_dataset(val_files, False)  # 25
    test_dataset = generator.make_dataset(test_files, False)  # 25
    tf.data.experimental.save(train_dataset, os.path.join(tf_dataset_stft_path, 'th_train'))
    tf.data.experimental.save(val_dataset, os.path.join(tf_dataset_stft_path, 'th_val'))
    tf.data.experimental.save(test_dataset, os.path.join(tf_dataset_stft_path, 'th_test'))

    return


def load_data_from_disk(tf_dataset_path, mfcc=True):
    if mfcc is True:
        tf_dataset_path = os.path.join(tf_dataset_path, "mfcc")
        tensor_specs = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
                        tf.TensorSpec([None], dtype=tf.int64)
                        )
    else:
        tf_dataset_path = os.path.join(tf_dataset_path, "stft")
        tensor_specs = (tf.TensorSpec([None, 32, 32, 1], dtype=tf.float32),
                        tf.TensorSpec([None], dtype=tf.int64)
                        )

    train_dataset = tf.data.experimental.load(os.path.join(tf_dataset_path, "th_train"), tensor_specs)
    val_dataset = tf.data.experimental.load(os.path.join(tf_dataset_path, "th_val"), tensor_specs)
    test_dataset = tf.data.experimental.load(os.path.join(tf_dataset_path, "th_test"), tensor_specs)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--version', type=str, default="big", choices=["big", "little"], help='Model version to train')
    args = parser.parse_args()
    main(args.version)