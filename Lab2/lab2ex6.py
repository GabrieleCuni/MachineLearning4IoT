#!/usr/bin/env python3 
from argparse import ArgumentParser
import tensorflow as tf
import time
import os

parser = ArgumentParser()
parser.add_argument("--input", type=str, default="out_5.wav", help="Input file name. Default: out_5.wav")
parser.add_argument("--output", type=str, default="out_6.png", help="Output file name. Default: out_6.png")
parser.add_argument("--length", type=float, default=0.04, help="frame length in s")
parser.add_argument("--stride", type=float, default=0.02, help="Stride in s")
args = parser.parse_args()

audio = tf.io.read_file(args.input)
tf_audio, rate = tf.audio.decode_wav(audio)
tf_audio = tf.squeeze(tf_audio, 1)
print(f"Sample shape:{tf_audio.shape}")

#  An integer scalar Tensor. The window length in samples. 
#  An integer scalar Tensor. The number of samples to step. 
frame_length = int(args.length * rate.numpy()) # Absolute number equal to the samples number.
frame_step = int(args.stride * rate.numpy())
print(f"Frame length: {frame_length}")
print(f"Frame step: {frame_step}")

t0 = time.time()
stft = tf.signal.stft(	tf_audio,\
			frame_length=frame_length,\
			frame_step=frame_step,\
			fft_length=frame_length)
t1 = time.time()
print(f"Spectrogram computation time {round(t1-t0,3)} s")
spectrogram = tf.abs(stft) # I compute the abs because stft is composed by complex numbers and I want its magnitude.
print(f"Spectrogram shape: {spectrogram.shape}")

spectrogram_byte = tf.io.serialize_tensor(spectrogram)
filename_byte = f"{os.path.splitext(args.input)[0]}.tf"
tf.io.write_file(filename_byte, spectrogram_byte)
size_i = os.path.getsize(args.input) / 2.**10
size_o = os.path.getsize(filename_byte) / 2.**10
print(f"Input size({args.input}): {round(size_i, 2)} KB")
print(f"Spectrogram size({filename_byte}): {round(size_o, 2)} KB")

image = tf.transpose(spectrogram)
image = tf.expand_dims(image, -1)
image = tf.math.log(image + 1.e-6) # tf.cast(x, dtype, name=None)
min_ = tf.reduce_min(image)
max_ = tf.reduce_max(image)
image = (image - min_) / (max_ - min_)
image = image * 255.
image = tf.cast(image, tf.uint8)
image = tf.io.encode_png(image)
tf.io.write_file(args.output, image)
print(f"File {args.output} is saved")
