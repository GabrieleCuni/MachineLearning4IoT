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
from mymodules.preprocessing import SignalGenerator

def make_mlp(units):
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=units)
    ])

    return model

def make_cnn(units, strides):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=units)
    ])

    return model

def make_ds_cnn(units, strides):
    model = keras.Sequential([
        keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
        keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
        keras.layers.BatchNormalization(momentum=0.1),
        keras.layers.ReLU(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(units=units)
    ])

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', type=str, default="mlp", help='model name')
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument('--mfcc', action='store_true', help='use MFCCs')
    parser.add_argument("-s", '--silence', action='store_true', help='add silence')
    parser.add_argument("-v", '--verbose', action='store_true')
    args = parser.parse_args()

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # zip_path = tf.keras.utils.get_file(
    #     origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    #     fname='mini_speech_commands.zip',
    #     extract=True,
    #     cache_dir='.', 
    #     cache_subdir='data')

    if args.silence:
        num_samples = 9000
        units = 9
        data_dir = os.path.join('.', 'data', 'mini_speech_commands_silence')
    else:
        num_samples = 8000
        units = 8
        data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    

    train_files = filenames[:int(num_samples*0.8)]
    val_files = filenames[int(num_samples*0.8): int(num_samples*0.9)]
    test_files = filenames[int(num_samples*0.9):]

    labels = np.array(tf.io.gfile.listdir(str(data_dir)))
    labels = labels[labels != 'README.md']

    STFT_OPTIONS = {'stft_frame_length': 256, 'stft_frame_step': 128, 'mfcc': False}
    MFCC_OPTIONS = {'stft_frame_length': 640, 'stft_frame_step': 320, 'mfcc': True,
                    'lower_freq_mel': 20, 'upper_freq_mel': 4000, 'num_mel_bins': 40,'num_coefficients': 10}
    
    if os.path.exists("./models") is False:
        os.mkdir("./models")
        os.mkdir("./models/silence")


    if args.mfcc is True:
        options = MFCC_OPTIONS
        strides = [2, 1]
        if args.model == "mlp":
            model = make_mlp(units)
            if args.silence:
                if os.path.exists('./models/silence/MLP_MFCC/') is False:
                    os.mkdir('./models/silence/MLP_MFCC/')
                filepath = './models/silence/MLP_MFCC/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
            else:
                if os.path.exists('./models/MLP_MFCC/') is False:
                    os.mkdir('./models/MLP_MFCC/')
                filepath = './models/MLP_MFCC/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
        elif args.model == "cnn":
            model = make_cnn(units, strides)
            if args.silence:
                if os.path.exists('./models/silence/CNN_MFCC/') is False:
                    os.mkdir('./models/silence/CNN_MFCC/')
                filepath = './models/silence/CNN_MFCC/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
            else:
                if os.path.exists('./models/CNN_MFCC/') is False:
                    os.mkdir('./models/CNN_MFCC/')
                filepath = './models/CNN_MFCC/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
        elif args.model == "dscnn":
            model = make_ds_cnn(units, strides)
            if args.silence:
                if os.path.exists('./models/silence/DSCNN_MFCC/') is False:
                    os.mkdir('./models/silence/DSCNN_MFCC/')
                filepath = './models/silence/DSCNN_MFCC/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'    
            else:
                if os.path.exists('./models/DSCNN_MFCC/') is False:
                    os.mkdir('./models/DSCNN_MFCC/')
                filepath = './models/DSCNN_MFCC/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
        else:
            print("Error: -m, --model [mlp, cnn, dscnn]")
    else:
        options = STFT_OPTIONS
        strides = [2, 2]
        if args.model == "mlp":
            model = make_mlp(units)
            if args.silence:
                if os.path.exists('./models/silence/MLP_STFT/') is False:
                    os.mkdir('./models/silence/MLP_STFT/')
                filepath = './models/silence/MLP_STFT/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'    
            else:
                if os.path.exists('./models/MLP_STFT/') is False:
                    os.mkdir('./models/MLP_STFT/')
                filepath = './models/MLP_STFT/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
        elif args.model == "cnn":
            model = make_cnn(units, strides)
            if args.silence:
                if os.path.exists('./models/silence/CNN_STFT/') is False:
                    os.mkdir('./models/silence/CNN_STFT/')
                filepath = './models/silence/CNN_STFT/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'    
            else:
                if os.path.exists('./models/CNN_STFT/') is False:
                    os.mkdir('./models/CNN_STFT/')
                filepath = './models/CNN_STFT/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
        elif args.model == "dscnn":
            model = make_ds_cnn(units, strides)
            if args.silence:
                if os.path.exists('./models/silence/DSCNN_STFT/') is False:
                    os.mkdir('./models/silence/DSCNN_STFT/')
                filepath = './models/silence/DSCNN_STFT/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
            else:
                if os.path.exists('./models/DSCNN_STFT/') is False:
                    os.mkdir('./models/DSCNN_STFT/')
                filepath = './models/DSCNN_STFT/model_{epoch:02d}_{val_sparse_categorical_accuracy:.4f}'
        else:
            print("Error: -m, --model [mlp, cnn, dscnn]")

    generator = SignalGenerator(labels, 16000, **options)

    train_dataset = generator.make_dataset(train_files, True)
    val_dataset = generator.make_dataset(val_files, False)
    test_dataset = generator.make_dataset(test_files, False)
       

    model.compile(  optimizer=tf.optimizers.Adam(),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=keras.metrics.SparseCategoricalAccuracy()
    )

    save_model = keras.callbacks.ModelCheckpoint(filepath=filepath, 
                                                monitor="val_sparse_categorical_accuracy",
                                                save_best_only=True,
                                                save_weights_only=False,
                                                save_freq='epoch'
    )

    model.fit(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=save_model)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
