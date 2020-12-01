import time
import argparse
from board import D4
import adafruit_dht
from datetime import datetime
import tensorflow.lite as tflite
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./models_5_1")
    args = parser.parse_args()


    samples_number = 6
    sleep_time = 1
    input_format = []

    sensor = adafruit_dht.DHT11(D4)

    for _ in range(samples_number):
        t = sensor.temperature
        h = sensor.humidity
        input_format.append([t, h])
        time.sleep(sleep_time)

    input_format = np.array(input_format, dtype=np.float32) # shape = (6,2)
    input_format = np.expand_dims(input_format, axis=0) # shape = (1,6,2)

    # Normalization:
    mean = [ 9.107597, 75.904076]
    std = [ 8.654227, 16.557089]
    input_format = (input_format - mean) / std
    input_format = input_format.astype(np.float32)

    for model_name in os.listdir(args.input):
        model_path = os.path.join(args.input, model_name)
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_format)
        interpreter.invoke()
        t0 = time.time()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        t1 = time.time()
        t, h = sensor.temperature, sensor.humidity

        print(model_name)
        print(f"Input: {input_format} - Prediction: {prediction} - True value: {t}, {h} - latency: {t1-t0:.2f} s")

if __name__ == "__main__":
    main()