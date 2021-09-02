import os
import sys
import pyaudio
from time import time
from scipy import signal
import numpy as np
import tensorflow as tf
import io
import subprocess
import argparse

# globals
channels = 1
chunk_size = 4800
record_seconds = 1
input_device_index = 0
resolution = pyaudio.paInt16
res_string = "16bit"
sampling_rate = 48000
resampling_rate = 16000
num_mel_bins = 40
lower_frequency = 20
upper_frequency = 4000
window_length_s = 0.04  # 40ms
stride_s = 0.02
frame_length = int(resampling_rate * window_length_s)  # STFT samples per frame
frame_step = int(resampling_rate * stride_s)  # STFT number of samples to stride
n_stream_reads = int(sampling_rate / chunk_size * record_seconds)
num_spectrogram_bins = int(resampling_rate * stride_s + 1)
timings = []
    
def main():
    parser = argparse.ArgumentParser(description="Iteratively samples 1-second audio signals,\
                                        process the MFCCs, and store the output on disk.\
                                        The script takes as input the number of samples to collect and\
                                        the output path where to store the MFCCs.")
    parser.add_argument("-n", "--num-samples", type=int, default=5, help="Samples number must be recorded. Default: 3")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output path. Default: ./output")
    args = parser.parse_args()

    n_samples = args.num_samples
    output_path = args.output

    # suppress pyaudio errors
    os.close(sys.stderr.fileno())
    # remove previous files
    [os.remove(os.path.join(output_path, filename)) for filename in [f"mfccs{n}.bin" for n in range(n_samples)] if os.path.exists(os.path.join(output_path, filename))]
    # compute mel weight matrix before for efficiency
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, resampling_rate, lower_frequency, upper_frequency)


    audio_handle = pyaudio.PyAudio()
    # open input stream
    stream = audio_handle.open(format=resolution, channels=channels, rate=sampling_rate, input=True, frames_per_buffer=chunk_size, input_device_index=input_device_index)
    stream.stop_stream()

    # switch to powersave and wait for it
    p = subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
    p.wait()

    # reset the monitors
    p = subprocess.Popen(["sudo", "/bin/sh", "-c", "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"]) 
    p.wait()

    for sample in range(n_samples):
        start = time()
        # RECORDING PHASE
        stream.start_stream()
        buffer = io.BytesIO()

        for i in range(0, n_stream_reads):
            if i == n_stream_reads-2:
                # switch to performance at the second last read for the preprocessing
                subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
            buffer.write(stream.read(chunk_size, exception_on_overflow=False))

        # stop input stream
        stream.stop_stream()

        # RESAMPLING PHASE
        audio = np.fromstring(buffer.getvalue(), dtype=np.int16)
        audio = signal.resample_poly(audio, 1, sampling_rate//resampling_rate).astype(np.int16)

        # SPECTROGRAM COMPUTATION PHASE
        tf_audio = tf.constant(audio.astype(np.float32))
        stft = tf.signal.stft(tf_audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
        spectrogram = tf.abs(stft)

        # MFCCs COMPUTATION PHASE
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10]

        # WRITING MFFCs TO DISK PHASE
        tf.io.write_file(os.path.join(output_path, f"mfccs{sample}.bin"), tf.io.serialize_tensor(mfccs))

        end = time()
        timings.append(end-start)

        # back to powersave
        subprocess.Popen(['sudo', '/bin/sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])

    stream.close()
    audio_handle.terminate()
    for timing in timings:
        print(round(timing, 3))

    # Read the monitors
    p = subprocess.Popen(["cat", "/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state"])
    p.wait()


if __name__ == '__main__':
    main()

