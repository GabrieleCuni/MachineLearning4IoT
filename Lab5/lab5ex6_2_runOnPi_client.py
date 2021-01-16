import argparse
import base64
import datetime
import numpy as np
import pyaudio
import requests
import wave
from io import BytesIO
from scipy import signal
from scipy.io import wavfile

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

chunk = 2400
resolution = pyaudio.paInt16
samp_rate = 48000
racord_secs = 1
chunks = int((samp_rate / chunk) * record_secs)

frames = []
audio = pyaudio.PyAudio()

now = datetime.datetime.now()
timestamp = int(now.timestamp())

stream = audio.open(format=resolution, rate=samp_rate, channels=1, input=True, frames_per_buffer=chunk)

for i in range(chunks):
    data = stream.read(chunk)
    frames.append(data) 
stream.stop_stream()

audio = np.frombuffer(b"".join(frames), dtype=np.int16)
audio = signal.resample_poly(audio, 1, 3)
audio = audio.astype(np.int16)
buf = BytesIO()
wavefile = wave.open(buf, "wb")
wavefile.setnchannels(1)
wavefile.setsampwidth(2)
wavefile.setframerate(16000)
wavefile.writeframes(audio.tobytes())
wavefile.close()
buf.seek(0)

audio_b64bytes = base64.b64encode(buf.read())
audio_string = audio_b64bytes.decode()

body = {
    "bn": "http://raspberrypi.local",
    "bt": timestamp,
    "e": [
        {"n":"audio", "u":"/", "t":0, "vd":audio_string}
    ]
}

url = "http://raspberrypi.local:8080/{}".format(args.model)
r = requests.put(url, json=body)

if r.status_code == 200:
    rbody = r.json()
    prob = rbody["probability"]
    label = rbody["label"]

    print("{} ({}%)".format(label, prob))
else:
    print("Error")
    print(r.text)

