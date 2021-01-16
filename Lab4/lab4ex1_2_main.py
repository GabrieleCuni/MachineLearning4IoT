import argparse
import os 
import numpy as np 
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--silence", action="store_true")
    args = parser.parse_args()

    dataset_path = os.path.join(".", "datasets")
    model_path = os.path.join(".", "tflite_models")

    if args.silence:
        dataset_path = os.path.join(dataset_path, "silence")
        model_path = os.path.join(model_path, "silence")

    if args.mfcc:
        dataset_path = os.path.join(dataset_path, "mfcc", "th_test")
        model_path = os.path.join(model_path, "mfcc")
        tensor_specs = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
                        tf.TensorSpec([None], dtype=tf.int64)
        )
    else:
        dataset_path = os.path.join(dataset_path, "stft", "th_test")
        model_path = os.path.join(model_path, "stft")
        tensor_specs = (tf.TensorSpec([None, 32, 32, 1], dtype=tf.float32),
                        tf.TensorSpec([None], dtype=tf.int64)
        )

    test_dataset = tf.data.experimental.load(dataset_path, tensor_specs)
    test_dataset = test_dataset.unbatch().batch(1)

    interpreter = tf.lite.Interpreter(model_path=os.path.join(model_path, f"{args.model}.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accuracy = 0
    count = 0
    for x, y_true in test_dataset:
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]["index"])
        y_pred = y_pred.squeeze()
        y_pred = np.argmax(y_pred)
        y_true = y_true.numpy().squeeze()
        accuracy += y_pred == y_true
        count += 1

    accuracy /= float(count)
    print(f"Accuracy {accuracy:.2f}")