from keras.models import load_model
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException
from PIL import Image
import urllib3
urllib3.disable_warnings()
import requests
import io
import time
import numpy as np
import paho.mqtt.client as paho
import json
import sys
import logging
import config as c


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

STATES = {
    0 : 'OFF',
    1 : 'ON'
}
CROP = c.IMAGE_CROP_XY + tuple(sum(e) for e in zip(c.IMAGE_CROP_XY,c.IMAGE_SIZE))
# autodiscovery topic format
TOPIC = 'homeassistant/binary_sensor/{0}/{1}'


def get_image():
    try:
        resp = requests.get(c.IMAGE_URL, auth=HTTPBasicAuth(c.USER, c.PASSWORD), verify=False)
        fp = io.BytesIO(resp.content)
        image = Image.open(fp)
        img_crop = image.crop(CROP)
        img = np.expand_dims(img_crop, axis=0)
        return img
    except RequestException:
        LOGGER.error(f'failed to fetch image from {c.IMAGE_URL}', exc_info=True)
    except Exception:
        LOGGER.error('image processing error', exc_info=True)
    return None

def main():
    LOGGER.info(f'trying to load model...')
    model = load_model(c.MODEL_NAME)
    LOGGER.info(f'model {c.MODEL_NAME} loaded')
    client = paho.Client()
    client.username_pw_set(c.MQTT_USER, c.MQTT_PASSWORD)
    client.connect(c.MQTT_HOST, c.MQTT_PORT)
    client.loop_start()
    LOGGER.info(f'connection to {c.MQTT_HOST}:{c.MQTT_PORT} established')
    config_topic = TOPIC.format(c.MQTT_DEVICE_ID, 'config')
    state_topic = TOPIC.format(c.MQTT_DEVICE_ID, 'state')
    sensor_config = {
        "name": c.MQTT_DEVICE_NAME,
        "device_class": "power",
        "state_topic": state_topic
    }
    val = json.dumps(sensor_config)
    client.publish(config_topic, val, qos=1, retain=True)

    prev_state = None
    state_change_count = 0

    while True:
        img = get_image()
        if img is not None:
            classes = model.predict_classes(img)
            state = STATES[int(classes[0])]
            if prev_state is None or state_change_count >= c.N_WAIT:
                client.publish(state_topic, state, qos=1, retain=True)
                prev_state = state
                state_change_count = 0
            else:
                if prev_state != state:
                    state_change_count += 1
                else:
                    state_change_count = 0
        time.sleep(1)


if __name__ == '__main__':
    main()