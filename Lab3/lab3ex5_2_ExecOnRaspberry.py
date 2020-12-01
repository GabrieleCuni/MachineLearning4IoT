import argparse
import os
import tensorflow.lite as tflite

import pyaudio
import numpy as np
import io
from scipy import signal
import wave
from scipy.io import wavfile
import tensorflow as tf


class AudioHandler:
    def __init__(   self, resolution=16, channels=1, sampling_rate=48000, 
                    chunk=4800, stft_frame_length=256, stft_frame_step=128,
                    num_mel_bins=40, lower_freq_mel=20, upper_freq_mel=4000,
                    num_coefficients=10):
        if resolution == 8:
            self.res = pyaudio.paInt8
        elif resolution == 16:
            self.res = pyaudio.paInt16
        elif resolution == 32:
            self.res = pyaudio.paInt32
        else:
            print("Error: resolution must be 8, 16 or 32")

        self._audio_handler = pyaudio.PyAudio()
        self._stream = self._audio_handler.open(format=self.res,
                                                channels=channels,
                                                rate=sampling_rate,
                                                frames_per_buffer=chunk,
                                                input=True
        )
        self._stream.stop_stream()

        self.sampling_rate = sampling_rate
        self.chunk = chunk
        self.channels = channels
        self.stft_frame_length = stft_frame_length
        self.stft_frame_step = stft_frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_freq_mel = lower_freq_mel
        self.upper_freq_mel = upper_freq_mel
        self.num_coefficients = num_coefficients

        self.num_spectrogram_bins = (self.stft_frame_length) // 2 + 1        
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, self.num_spectrogram_bins, self.sampling_rate,
                self.lower_freq_mel, self.upper_freq_mel
            )

    def recording(self, record_seconds=1):
        n_stream_reads = int(self.sampling_rate / self.chunk * record_seconds)
        buffer = io.BytesIO()
        frames = []

        self._stream.start_stream()
        for _ in range(n_stream_reads):
            data = self._stream.read(self.chunk)
            frames.append(data)
        self._stream.stop_stream()

        wavefile = wave.open(buffer, "wb")
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._audio_handler.get_sample_size(self.res))
        wavefile.setframerate(self.sampling_rate)
        wavefile.writeframes(b"".join(frames))
        wavefile.close()
        buffer.seek(0)

        audio, _ = tf.audio.decode_wav(buffer.read())
        audio = tf.squeeze(audio, axis=1)
        
        return audio

    def resampling(self, audio, resampling_rate=16000):
        audio = signal.resample_poly(audio, 1, self.sampling_rate//resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        return audio

    def save_wav(self, audio, file_name):
        waveFile = wave.open(file_name, "wb")
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self._audio_handler.get_sample_size(self.res))
        waveFile.setframerate(self.sampling_rate)
        waveFile.writeframes(audio)
        waveFile.close()

        return

    def close(self):
        self._stream.close()
        self._audio_handler.terminate()

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(  audio,
                                frame_length=self.stft_frame_length,
                                frame_step=self.stft_frame_step,
                                fft_length=self.stft_frame_length
                            )
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

def padding(audio):
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([16000])

    return audio

def stft_preprocessing(audio):
    audio = padding(audio)
	# stft preprocess
    stft = tf.signal.stft(audio,frame_length=256,frame_step=128,fft_length=256)
    spectrogram = tf.abs(stft)
    spectrogram = tf.expand_dims(spectrogram, -1)
    # spectrogram = tf.reshape(spectrogram, [1, 124, 129, 1])
    spectrogram = tf.image.resize(spectrogram, [32, 32])

    return spectrogram

def mfcc_preprocessing(audio):
    audio = padding(audio)
    # mfcc preprocess
    num_mel_bins = 40
    num_spectrogram_bins = (640) // 2 + 1 
    sampling_rate = 48000
    lower_freq_mel = 20
    upper_freq_mel = 4000
    num_coefficients = 10
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, sampling_rate,
                lower_freq_mel, upper_freq_mel
    )

    stft = tf.signal.stft(audio,frame_length=640,frame_step=320,fft_length=640)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.expand_dims(mfccs, -1)

    return mfccs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./models_5_2")
    args = parser.parse_args()    

    labels_silence = ['down', 'right', 'left', 'up', 'yes', 'no', 'go', 'silence', 'stop']    
    labels = ['down', 'right', 'left', 'up', 'yes', 'no', 'go', 'stop']

    print("Start recording")
    audioHandler = AudioHandler(stft_frame_length=256, stft_frame_step=128)
    audio = audioHandler.recording()
    print("Stop recording")
    audio = audioHandler.resampling(audio)
    audioHandler.close()
    # _, audio = wavfile.read("0ab3b47d_nohash_0.wav")
    # _, audio = wavfile.read("out_5.wav")

    stft_input = stft_preprocessing(audio)
    mfcc_input = mfcc_preprocessing(audio)
    # print(mfcc_input.shape)

    stft_input = tf.expand_dims(stft_input, axis=0)
    mfcc_input = tf.expand_dims(mfcc_input, axis=0)
    # print(mfcc_input.shape)

    for model_name in os.listdir(args.input):
        interpreter = tflite.Interpreter(model_path=os.path.join(args.input, model_name))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # print(input_details, "\n")

        if "STFT" in model_name:
            interpreter.set_tensor(input_details[0]['index'], stft_input) 
        else:
            interpreter.set_tensor(input_details[0]['index'], mfcc_input)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        if "silence" in model_name:
            # print(f"labels_silence len: {len(labels_silence)}")
            # print(f"Prediction index: {np.argmax(prediction)}")
            prediction = labels_silence[np.argmax(prediction)]
        else:
            # print(f"labels_silence len: {len(labels_silence)}")
            # print(f"Prediction index: {np.argmax(prediction)}")
            prediction = labels[np.argmax(prediction)]

        print(f"Model: {model_name} - Prediction: {prediction}")
        

if __name__ == "__main__":
    main()