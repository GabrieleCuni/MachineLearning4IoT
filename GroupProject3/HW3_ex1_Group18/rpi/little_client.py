import tensorflow as tf
import os
import numpy as np
import json
import requests
import base64
import sys
from preprocessing import SignalGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    tflite_model = os.path.join(".", "little.tflite")
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
    generator = SignalGenerator(labels, 16000, **STFT_OPTIONS)    

    if os.path.exists(tflite_model) is False:
        print(f"Error: {tflite_model} file not found!")
        sys.exit(1)    
    
    accuracy, communication_cost = test_little_big(tflite_model, test_files, generator)

    print(f'Accuracy: {accuracy}%')
    print(f'Communication cost: {communication_cost/(2**20):.4f} MB')


def test_little_big(tflite_model, test_files, generator):
    communication_cost = 0.0
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accuracy, count = 0, 0
    big_model_calls = 0
    for i, audio_path in enumerate(test_files):
        features, labels = generator._preprocess(audio_path)
        features = tf.expand_dims(features, axis=0)
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predictions = predictions.squeeze()
        predictions = tf.nn.softmax(tf.convert_to_tensor(predictions)).numpy()
        second_prediction, first_prediction = np.argsort(predictions)[-2:]  # get the 2 highest
        first_prob, second_prob = predictions[[first_prediction, second_prediction]]

        audio_binary = tf.io.read_file(audio_path)
        final_prediction, senml_size = success_checker(first_prob, second_prob, first_prediction, audio_binary)

        labels = labels.numpy().squeeze()
        accuracy += final_prediction == labels
        count += 1
        communication_cost += senml_size
        if senml_size > 0:
            big_model_calls += 1
    # print(f"\n{big_model_calls} big model calls.")
    return accuracy / float(count), communication_cost

def success_checker(most_prob, less_prob, prediction_most_arg, audio_binary):
    if float(most_prob - less_prob) < 0.50:
        # request for big model
        audio_b64bytes = base64.b64encode(audio_binary.numpy())
        audio_string = audio_b64bytes.decode()

        body = {"bn": "http://0.0.0.0:8080/",
                "e": [{"n": "audio", "u": "/", "t": "0", "vd": audio_string}]}

        url = "http://localhost:8080"
        r = requests.put(url, json=body)       

        if r.status_code == 200:
            rbody = r.json()
            prediction = int(rbody['label'])
            senml_string = json.dumps(body)   
            size_senml = len(senml_string)
            return prediction, size_senml
        else:
            print(f'Error in big request: {r.status_code}')
            print(r.text)
            sys.exit(1)
    else:
        return prediction_most_arg, 0.0


if __name__ == "__main__":
    main()