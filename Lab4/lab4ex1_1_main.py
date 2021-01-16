import argparse
import numpy as np 
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--silence", action="store_true")
    args = parser.parse_args()

    # if args.mfcc:
    #     dataset_name = "./"
    #     tensor_specs = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
    #                     tf.TensorSpec([None], dtype=tf.int64)
    #     )
    # else:
    #     dataset_name = "./"
    #     tensor_specs = (tf.TensorSpec([None, 32, 32, 1], dtype=tf.float32),
    #                     tf.TensorSpec([None], dtype=tf.int64)
    #     )

    dataset_name = "./th_test"
    tensor_specs = (tf.TensorSpec([None, 49, 10, 1], dtype=tf.float32),
                    tf.TensorSpec([None], dtype=tf.int64)
    )

    test_dataset = tf.data.experimental.load(dataset_name, tensor_specs)
    test_dataset = test_dataset.unbatch().batch(1)

    interpreter = tf.lite.Interpreter(model_path="sgsssa")
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

    

    

if __name__ == "__main__":
    main()

