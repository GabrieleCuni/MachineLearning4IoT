import argparse
import base64
import json
import time
from datetime import datetime
import tensorflow as tf
import numpy as np

from DoSomething import DoSomething

class Inference(DoSomething):
    def __init__(self, clientID, features_type):
        super().__init__(clientID)
        self._interpreter = tf.lite.Interpreter(model_path=f'{clientID}.tflite')
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        if features_type == 'MFCC':
            self._feature_shape = (1, 49, 10, 1)
        else:
            self._feature_shape = (1, 32, 32, 1)

    def _infer(self, features):
        self._interpreter.set_tensor(self._input_details[0]['index'], features)
        self._interpreter.invoke()
        prediction = self._interpreter.get_tensor(self._output_details[0]['index'])
        prediction = prediction.squeeze()
        return prediction

    def notify(self, topic, msg):
        print(topic)
        audio_index = topic.split("/")[4]
        senml = json.loads(msg)
        event = senml["e"]
        features_string = event[0]["vd"]
        features_bytes = base64.b64decode(features_string)
        mffc_features = np.reshape(np.frombuffer(features_bytes, dtype=np.float32), self._feature_shape)

        prediction = self._infer(mffc_features)

        prediction_b64bytes = base64.b64encode(prediction)
        prediction_string = prediction_b64bytes.decode()

        name_type = 'probabilities' if self.clientID == '2' else 'logits'

        body = {
            "bn": f"http://192.168.1.14:8080/{self.clientID}",
            "bt": int(datetime.now().timestamp()),
            "e": [{"n": f"{name_type}", "u": "/", "t": 0, "vd": prediction_string}]
        }

        output = json.dumps(body)
        self.myMqttClient.myPublish(f"/ts3tdhj4djcygj5yytdfs764regrg/inference/{audio_index}", output)
        

def main(args=None):
    model_features_mapping = {1: 'MFCC', 2: 'MFCC', 3: 'MFCC', 4: 'STFT', 5: 'STFT'}
    inferece = Inference(str(args.model), model_features_mapping[args.model])
    inferece.run()
    inferece.myMqttClient.mySubscribe(f"/ts3tdhj4djcygj5yytdfs764regrg/coopclient/{model_features_mapping[args.model]}/+")

    while True:
        time.sleep(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=int, choices=[1, 2, 3, 4, 5], required=True)
    args = parser.parse_args()
    main(args)