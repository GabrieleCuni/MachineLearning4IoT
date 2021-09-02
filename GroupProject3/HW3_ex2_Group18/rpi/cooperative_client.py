import argparse
import os
import base64
import json
import time
import numpy as np
import tensorflow as tf 
from datetime import datetime
from collections import defaultdict

from DoSomething import DoSomething
from preprocessing import SignalGenerator
import sys


class Cooperative(DoSomething):
    def __init__(self, clientID):
        super().__init__(clientID)
        self._true_labels = []
        self._probabilities = defaultdict(list)
        self._complete = np.full(800, False, dtype=np.bool)
        self._cooperative_inference_policy = self.majority_voting
        self._n_inference_clients = 5

    def feature_publisher(self, test_files, generator_STFT, generator_MFCC):
        for i, audio_path in enumerate(test_files):

            features_STFT, label = generator_STFT._preprocess(audio_path)
            self._true_labels.append(label)
            features_MFCC, _ = generator_MFCC._preprocess(audio_path)

            features_STFT_b64bytes = base64.b64encode(features_STFT.numpy())
            features_MFCC_b64bytes = base64.b64encode(features_MFCC.numpy())

            features_STFT_string = features_STFT_b64bytes.decode()
            features_MFCC_string = features_MFCC_b64bytes.decode()

            body_STFT = {
                "bn": f"http://192.168.1.2:8080/{self.clientID}",
                "bt": int(datetime.now().timestamp()),
                "e": [{"n": "feature", "u": "/", "t": 0, "vd": features_STFT_string}]
            }
            body_MFCC = {
                "bn": f"http://192.168.1.2:8080/{self.clientID}",
                "bt": int(datetime.now().timestamp()),
                "e": [{"n": "feature", "u": "/", "t": 0, "vd": features_MFCC_string}]
            }

            output = json.dumps(body_STFT)
            self.myMqttClient.myPublish(f"/ts3tdhj4djcygj5yytdfs764regrg/coopclient/STFT/{i}", output)
            output = json.dumps(body_MFCC)
            self.myMqttClient.myPublish(f"/ts3tdhj4djcygj5yytdfs764regrg/coopclient/MFCC/{i}", output)

            time.sleep(0.1)


    def notify(self, topic, msg):
        audio_index = int(topic.split("/")[-1])
        senml = json.loads(msg)
        event = senml["e"]
        resource_string = event[0]["vd"]
        resource_bytes = base64.b64decode(resource_string)
        resource = np.frombuffer(resource_bytes, dtype=np.float32)
        if event[0]['n'] == 'logits':
            resource = tf.nn.softmax(tf.convert_to_tensor(resource)).numpy()

        if not True in [np.all(resource == probs) for probs in self._probabilities[audio_index]]:
            self._probabilities[audio_index].append(resource)
        if len(self._probabilities[audio_index]) == self._n_inference_clients:
            self._complete[audio_index] = True
            print(f"\r {np.sum(self._complete)}/800 files complete", end='')
        if np.all(self._complete):
            final_accuracy = self._test_accuracy()
            print(f"Final accuracy: {final_accuracy}")
            self.end()
            sys.exit(0)


    def _test_accuracy(self):
        accuracy = 0
        for index, probs in self._probabilities.items():
            if self._true_labels[index] == self._cooperative_inference_policy(probs):
                accuracy += 1
        return accuracy / float(len(self._probabilities))


    def majority_voting(self, probs_list):
        predictions = [np.argmax(probs) for probs in probs_list]
        unique, counts = np.unique(predictions, return_counts=True)
        if len(unique) == 5:
            vote = np.argmax(np.array([max(probs) for probs in probs_list]))
        else:
            vote = unique[np.argmax(counts)]

        return vote


def main():
    data_path = os.path.join("data")
    dataset_path = os.path.join(data_path, "mini_speech_commands")
    if not os.path.exists(dataset_path):
        tf.keras.utils.get_file(origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                fname='mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir=data_path)
    with open("./kws_test_split.txt", "r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  # len 800
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
    generator_STFT = SignalGenerator(labels, 16000, **STFT_OPTIONS)
    generator_MFCC = SignalGenerator(labels, 16000, **MFCC_OPTIONS)

    cooperative = Cooperative("Group18_CooperativeClient")
    cooperative.run()

    cooperative.myMqttClient.mySubscribe("/ts3tdhj4djcygj5yytdfs764regrg/inference/+")

    cooperative.feature_publisher(test_files, generator_STFT, generator_MFCC)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()