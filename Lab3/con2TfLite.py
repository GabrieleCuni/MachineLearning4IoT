import argparse
import os 
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, help="Output file name without extention")
parser.add_argument("-i", "--input", type=str, help="Input directory name")
parser.add_argument("-s", "--silence", action="store_true")
args = parser.parse_args()

filepath = os.path.join(".", "tflite_modelsEx3")

if os.path.exists(args.input) is False:
    print("Error: Input file path does not exist")
    sys.exit(1)
else:
    if os.path.exists(filepath) is False:
        os.makedirs(filepath)

    converter = tf.lite.TFLiteConverter.from_saved_model(args.input)
    tflite_model = converter.convert()

    if args.silence:
        filename = os.path.join(filepath, f"{args.output}_silence.tflite")
    else:
        filename = os.path.join(filepath, f"{args.output}.tflite")

    with open(filename, 'wb') as fp:
        fp.write(tflite_model)


