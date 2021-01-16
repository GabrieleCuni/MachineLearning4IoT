import cherrypy
import json
import base64
from board import D4
import adafruit_dht
import datetime
import pyaudio

class Sensors():
    exposed = True

    def __init__(self):
        super().__init__()
        self.dht_device = adafruit_dht.DHT11(D4)
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16, rate=48000, channels=1, input=True, frames_per_buffer=4800)
        self.stream.stop_stream()
        return

    def GET(self, *path, **query):        
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        temperature = self.dht_device.temperature
        humidity = self.dht_device.humidity
        frames = []
        self.stream.start_stream()
        i = 0
        while i<10:
            data = self.stream.read(4800)
            frames.append(data)
            i += 1
        self.stream.stop_stream()
        audio_b64bytes = base64.b64encode(b"".join(frames))
        audio_string = audio_b64bytes.decode()

        body = {
            "bn": "http://raspberrypi.local",
            "bt": timestamp,
            "e": [
                {"n": "temperature", "u": "Cel", "t": 0, "v": temperature},
                {"n": "humidity", "u": "RH", "t": 0, "v": humidity},
                {"n": "audio", "u": "/", "t":0, "vd": audio_string}
            ]
        }
        body = json.dumps(body)
        return body

        

    def PUT(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

def main():
    conf = {"/":{"request.dispatch": cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Sensors(), "", conf)
    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    cherrypy.config.update({"server.socket_port": 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()    


if __name__ == "__main__":
    main()