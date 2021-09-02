import base64
import cherrypy
import tensorflow as tf
import json
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class InferenceService:
    exposed = True

    def __init__(self, model_path='big.tflite'):
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(40, 321, 16000, 20, 4000)

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        events = input_body['e']

        audio_string = None
        for event in events:
            if event['n'] == 'audio':
                audio_string = event['vd']
        if audio_string is None:
            raise cherrypy.HTTPError(400, 'No audio event')

        audio_bytes = base64.b64decode(audio_string)
        mfcc_features = self._preprocess(audio_bytes)

        prediction = self._infer(mfcc_features)

        output_body = json.dumps({'label': str(prediction)})
        return output_body

    def _preprocess(self, audio_bytes):
        # decode and normalize
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)

        # padding
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([16000])

        # compute STFT
        stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=640)
        spectrogram = tf.abs(stft)

        # compute MFCC
        mel_spectrogram = tf.tensordot(spectrogram, self._linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfcc_features = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfcc_features = mfcc_features[..., :10]

        mfcc_features = tf.expand_dims(mfcc_features, -1)  # add channel dimension
        mfcc_features = tf.expand_dims(mfcc_features, 0)  # add batch dimension

        return mfcc_features

    def _infer(self, features):
        self._interpreter.set_tensor(self._input_details[0]['index'], features)
        self._interpreter.invoke()
        prediction = self._interpreter.get_tensor(self._output_details[0]['index'])
        prediction = prediction.squeeze()
        prediction = np.argmax(prediction)
        return prediction


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                  'tools.sessions.on': True}}

    cherrypy.tree.mount(InferenceService(), '/', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()