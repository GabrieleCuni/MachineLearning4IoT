import argparse
import pyaudio
import wave
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, default=5)
parser.add_argument("-r", type=int, default=16)
parser.add_argument("-s", type=float, default=48000)
parser.add_argument("-o", type=str, default="output_ex5.wav")
args = parser.parse_args()

CHUNK = 1024
CHANNELS = 1

# instantiate PyAudio 
p = pyaudio.PyAudio()

#for i in range(p.get_device_count()):
#    print(p.get_device_info_by_index(i))

# open stream
format = {8:pyaudio.paInt8 ,16:pyaudio.paInt16, 32:pyaudio.paInt32}
RATE = args.s
stream = p.open(format=format[args.r], channels=CHANNELS, rate=args.s, input=True, frames_per_buffer=CHUNK)
print("recording..")
frames = []
t0 = time.time()
for i in range(0, int(args.s / CHUNK * args.t)):
	data = stream.read(CHUNK)
	frames.append(data)
t1 = time.time() - t0
print(f"finished recording in {t1} sec")

# Terminate stream
stream.stop_stream()
stream.close()
p.terminate()

# Write on file
t0 = time.time()
print("Writing on disk...")

waveFile = wave.open(args.o, "wb")
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(p.get_sample_size(format[args.r]))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

t1 = time.time() - t0
print(f"finished writing in {t1} sec")

size = os.path.getsize(args.o)
print(f"Output file: {args.o} ({size/1000} Kb)")
