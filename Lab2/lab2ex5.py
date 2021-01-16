#!/usr/bin/env python3
from scipy.io import wavfile
from scipy import signal
import time
import os
import argparse
import numpy as np

"""
After compression the audio file is still understandable, its size is reduce by 1/3 and the execution time is very low.
The compression technique is realy good, it can be executed on the edge IoT device without problems. 
Example input/output:    
    Playing WAVE 'in_6.wav' : Signed 16 bit Little Endian, Rate 48000 Hz, Mono
    Playing WAVE 'out_5.wav': Signed 16 bit Little Endian, Rate 16000 Hz, Mono

"""

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="in.wav", help="Input file name. Default: in.wav")
parser.add_argument("--output", default="out_5.wav", help="Output file name. Default: out.wav ")
parser.add_argument("--rate", type=int, default=16000, help="Output Sampling Ratio. Default: 16000")
args = parser.parse_args()


print(f"Start resampling...")
input_rate, audio = wavfile.read(args.input)

# The resulting sample rate is up / down times the original sample rate. 
# => input_rate * up/down | up = 1

sampling_ratio = input_rate / args.rate # 48000 / 16000 = 3

t0 = time.time()
audio = signal.resample_poly(audio, 1, sampling_ratio) # sampling_ratio = 3
t1 = time.time()
print(f"Resempling time: {round(t1-t0, 3)} s")

audio = audio.astype(np.int16)
wavfile.write(args.output, args.rate, audio)

original_size = round(os.path.getsize(args.input) / 2.**10, 2)
new_size = round(os.path.getsize(args.output) / 2.**10, 2)
print(f"Input size ({args.input}): {original_size} KB")
print(f"Output size ({args.output}): {new_size} KB") 
