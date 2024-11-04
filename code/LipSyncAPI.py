import os
from pathlib import Path

from flask import Flask, render_template
from flask_socketio import SocketIO

from mfcc_extractor_lib import setup_logger
from lipsync_pytorch import run_lipsync

VISEME_FOLDER = Path(__file__).resolve().parent / "static/viseme_images"

logger = setup_logger(script_name=os.path.splitext(os.path.basename(__file__))[0])


image_mapping = {
    0: f"{VISEME_FOLDER}/A.png",
    1: f"{VISEME_FOLDER}/B.png",
    2: f"{VISEME_FOLDER}/C.png",
    3: f"{VISEME_FOLDER}/D.png",
    4: f"{VISEME_FOLDER}/E.png",
    5: f"{VISEME_FOLDER}/F.png",
    6: f"{VISEME_FOLDER}/G.png",
    7: f"{VISEME_FOLDER}/H.png",
    8: f"{VISEME_FOLDER}/X.png",
}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('lipsync_anim.html')


def send_prediction():

    timeout_counter = 0

    for viseme in run_lipsync():
        actual_viseme = image_mapping.get(viseme)
        viseme_image_url = f"/static/viseme_images/{actual_viseme.split('/')[-1]}"

        logger.info(
            f"Emitting viseme prediction. Viseme: {viseme}, Image URL: {viseme_image_url}")
        try:
            socketio.emit('new_prediction', {'image_url': viseme_image_url})
            socketio.sleep(0.1)
        except TimeoutError as e:

            if timeout_counter <= 3:
                logger.warning(f"Operation timed out: {e}")
                timeout_counter += 1
            else:
                logger.error(f"Operation timed out: {e}")


@socketio.on('connect')
def handle_connect():

    logger.info("Connecting and starting emission")
    socketio.start_background_task(send_prediction)


if __name__ == '__main__':

    logger.info("Starting app...")
    socketio.run(app)
