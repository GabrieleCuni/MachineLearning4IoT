#!/usr/bin/env python3 
"""
python3 lab2ex7.py --input yes1.tf --mel 40 --coef 10
Spectrogram shape: (49, 321)
Execution time: 0.039s
MFCCs shape: (49,10)
Input size: 61.46 KB
MFCCs Size: 1.93 KB
"""

import argparse
import os
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./in/yes1.tf", help="File input name")
parser.add_argument("--mel", type=int, default=40, help="Mel bins")
parser.add_argument("--coef", type=int, default=10, help="MFCCs number")
args = parser.parse_args()

spectr = tf.io.read_file(args.input)
spectr = tf.io.parse_tensor(spectr, out_type=tf.float32)
print(f"Spectrogram shape: {spectr.shape}")

num_spectr_bins = spectr.shape[-1]
num_mel_bins = args.mel
sampling_rate = 10000

ts = time.time()
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectr_bins, sampling_rate, 20, 4000)
mel_spectrogram = tf.tensordot(spectr, linear_to_mel_weight_matrix, 1)
log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
mfccs = mfccs[..., :args.coef]
te = time.time()
print(f"Execution time: {te-ts:.3f}")
print(f"MFCCs shape: {mfccs.shape}")

mfccs_byte = tf.io.serialize_tensor(mfccs)
filename_byte = f"{os.path.splitext(args.input)[0]}_mfccs.tf"
tf.io.write_file(filename_byte, mfccs_byte)
input_size = os.path.getsize(args.input) / 2.**10
mfccs_size = os.path.getsize(filename_byte) / 2.**10
print(f"Input size: {input_size:.2f} KB")
print(f"MFCCS size: {mfccs_size:.2f} KB")

image = tf.transpose(mfccs)
image = tf.expand_dims(image, -1)
min_ = tf.reduce_min(image)
max_ = tf.reduce_max(image)
image = (image-min_) / (max_ - min_)
image = image * 255.
image = tf.cast(image, tf.uint8)
image = tf.io.encode_png(image)
filename_image = f"{os.path.splitext(args.input)[0]}_mfccs.png"
tf.io.write_file(filename_image, image)