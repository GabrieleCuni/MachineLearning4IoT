import DoSomething import DoSomething
import base64
import datetime
import time
import json
import wave

class CommandClient(DoSomething):
    def notify(self, topic, msg):
        senml = json.loads(msg)
        timestamp = senml["bt"]
        events = senml["e"]

        for event in events:
            if event["n"] == "audio":
                audio_string = event["vd"]
            else:
                raise RuntimeError("No audio event")

        date = datetime.datetime.fromtimestamp(timestamp)
        audio_bytes = base64.b64decode(audio_string)
        wav_output_filename = "{}.wav".format(timestamp)
        wavefile = wave.open(wav_output_filename, "wb")
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(48000)
        wavefile.writeframes(audio_bytes)
        wavefile.close()

if __name__ == "__main__":
    test = CommandClient("command")
    test.run()
    test.myMqttClient.mySubscribe("/206803/audio")

    now = datetime.datetime.now()
    timestamp = int(now.timestamp())
    senml = {
        "bn": "http:/169.254.37.210",
        "bt": timestamp,
        "e": [
            {"n": "record", "u": "/", "t":0, "vb":True}
        ]
    }

    senml = json.dumps(senml)

    test.myMqttClient.myPublish("/206803/record", senml)

    while True:
        time.sleep(1)