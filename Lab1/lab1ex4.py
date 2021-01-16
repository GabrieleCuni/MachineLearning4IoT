import time
import argparse
from board import D4
import adafruit_dht
from datetime import datetime

# datetime.now() outputs 2020-10-20 14:35:13.662193
 

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", type=float, default=5)	
	parser.add_argument("-p", type=float, default=20)
	parser.add_argument("-o", type=str, default="output_ex4.log")
	args = parser.parse_args()

	sampleNo = int(args.p / args.f)

	dht_device = adafruit_dht.DHT11(D4)

	with open(args.o, "w") as fp:
		for i in range(sampleNo):
			now = datetime.now()
			temperature = dht_device.temperature
			humidity = dht_device.humidity
			format = f"{now.day}/{now.month}/{now.year},{now.hour}:{now.minute}:{now.second},{temperature},{humidity}\n"
			print(format)
			fp.write(format)
			time.sleep(args.f)

if __name__=="__main__":
    main()
