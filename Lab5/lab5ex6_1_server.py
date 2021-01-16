import base64
import cherrypy
import tensorflow as tf 
import json

class InferenceService:
    exposed = True

    def __init__(self):
        mlp = tf.keras.models.load_model(".models/kws_mlp_True")
        cnn = tf.keras.models.load_model(".models/kws_cnn_True")
        dscnn = tf.keras.models.load_model(".models/kws_dscnn_True")
        self.labels = ["stop", "up", "yes", "right", "left", "no", "silence", "down", "go"]
        self.models = {
            "mlp": mlp,
            "cnn": cnn,
            "dscnn": dscnn
        }
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(40,321,16000,20,4000)

    def preprocess(self, audio_bytes):
        # Decode and Normalize
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)
        # STFT
        stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=640)
        spectrogram = tf.abs(stft)
        # MFCC
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :10]
        # Add channel dimension
        mfccs = tf.expand_dims(mfccs, -1)
        # Add batch dimension
        mfccs = tf.expand_dims(mfccs, 0)
        return mfccs

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        events = input_body["e"]

        audio_string = None
        for event in events:
            if event["n"] == "audio":
                audio_string = event["vd"]
        
        if audio_string is None:
            raise cherrypy.HTTPError(400, "No audio event")
        
        audio_bytes = base64.b64decode(audio_string)
        mfccs = self.preprocess(audio_bytes)

        model = self.models.get(path[0])

        if model is None:
            raise cherrypy.HTTPError(400, "No valid model")

        logits = model.predict(mfccs)
        probs = tf.nn.softmax(logits)
        prob = tf.reduce_max(probs).numpy() * 100
        label_idx = tf.argmax(probs, 1).numpy()[0]
        label = self.labels[label_idx]

        output_body = {
            "label": label,
            "probability": prob
        }

        output_body = json.dumps(output_body)

        return output_body