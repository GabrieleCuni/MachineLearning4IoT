import argparse
import os
import time
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="img.jpeg", help="Input Image")
parser.add_argument("--crop", type=int, default=168, help="crop size")
parser.add_argument("--resize", type=int, default=224, help="Output size")
args = parser.parse_args()

image = tf.io.read_file(args.input)
image = tf.io.decode_jpeg(image)

offset_height = (image.shape[0] - args.crop) // 2
offset_width = (image.shape[1] - args.crop) // 2

cropped = tf.image.crop_to_bounding_ox(image, offset_height, offset_width, args.crop, args.crop)

ts = time.time()
bilinear = tf.image.resize(cropped, [args.resize, args.resize], method="bilinear")
te = time.time()
print(f"Bilinear: {te-ts:.3f}")
filename = f"{os.path.splitext(args.input)[0]}_bilinear.jpeg"
bilinear = tf.cast(bilinear, tf.uint8)
bilinear = tf.image.encode_jpeg(bilinear)
tf.io.write_file(filename, bilinear)

ts = time.time()
bicubic = f.image.resize(cropped, [args.resize, args.resize], method="bicubic")
te = time.time()
print(f"Bicubic: {te-ts:.3f}")
filename = f"{os.path.splitext(args.input)[0]}_bicubic.jpeg"
bilinear = tf.cast(bicubic, tf.uint8)
bilinear = tf.image.encode_jpeg(bicubic)
tf.io.write_file(filename, bicubic)