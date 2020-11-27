import numpy as np
import os
import pandas as pd
import tensorflow as tf

class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std, verbose):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])
        self.verbose = verbose

    def split_window(self, features):
        # input_indeces = np.arange(self.input_width)
        inputs = features[:, :-1, :]

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        # It returns a tf.data.Dataset instance
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+1,
                sequence_stride=1,
                batch_size=32)
        # Maps map_func across the elements of this dataset.
        dataset = dataset.map(map_func=self.preprocess)
        if self.verbose:
            print(f"Dataset element: {dataset.element_spec}")
        # Caches the elements in this dataset.
        # The first time the dataset is iterated over, its elements will be cached either 
        # in the specified file or in memory. Subsequent iterations will use the cached data.
        dataset = dataset.cache() 
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset


class SignalGenerator():
    def __init__(self, 
            labels,     
            sampling_rate, 
            stft_frame_length, 
            stft_frame_step, 
            num_mel_bins=None, 
            lower_freq_mel=None,
            upper_freq_mel=None, 
            num_coefficients=None, 
            mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.stft_frame_length = stft_frame_length
        self.stft_frame_step = stft_frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_freq_mel = lower_freq_mel
        self.upper_freq_mel = upper_freq_mel
        self.num_coefficients = num_coefficients
        self.mfcc = mfcc

        num_spectrogram_bins = (self.stft_frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_freq_mel, self.upper_freq_mel
                )
            self._preprocess = self._preprocess_with_mfcc
        else:
            self._preprocess = self._preprocess_with_stft

    def _load_data(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def _padding(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def _get_spectrogram(self, audio):
        stft = tf.signal.stft(  audio,
                                frame_length=self.stft_frame_length,
                                frame_step=self.stft_frame_step,
                                fft_length=self.stft_frame_length
                            )
        spectrogram = tf.abs(stft)

        return spectrogram

    def _get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def _preprocess_with_stft(self, file_path):
        audio, label = self._load_data(file_path)
        audio = self._padding(audio)
        spectrogram = self._get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def _preprocess_with_mfcc(self, file_path):
        audio, label = self._load_data(file_path)
        audio = self._padding(audio)
        spectrogram = self._get_spectrogram(audio)
        mfccs = self._get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(self._preprocess)
        dataset = dataset.batch(32)
        dataset = dataset.cache()
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset

def test():
    pass

if __name__ == "__main__":
    test()