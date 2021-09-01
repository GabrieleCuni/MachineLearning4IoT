from DoSomething import DoSomething
import time
import json
from datetime import datetime

if __name__ == "__main__":
    test = DoSomething("Publisher 1")
    test.run()

    ten = True
    for i in range(10):
        now = datetime.now()
        
        datetime_str = now.strftime("%d-%m-%y %H:%M:%S")
        datetime_json = json.dumps({"datetime": datetime_str})

        test.myMqttClient.myPublish("/time/datetime", datetime_json)

        if ten is False:
            ten = True
        elif ten is True:
            ten = False
            timestamp = str((now.timestamp()))
            timestamp_json = json.dumps({"timestamp": timestamp})
            test.myMqttClient.myPublish("/time/timestamp", timestamp_json)

        time.sleep(5)

    test.end()