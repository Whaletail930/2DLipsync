from pathlib import Path

from flask import Flask, render_template
from flask_socketio import SocketIO

from lipsync_pytorch import run_lipsync

VISEME_FOLDER = Path(__file__).resolve().parent / "static/viseme_images"

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


def send_predictions():

    for viseme in run_lipsync():
        actual_viseme = image_mapping.get(viseme)
        viseme_image_url = f"/static/viseme_images/{actual_viseme.split('/')[-1]}"

        socketio.emit('new_prediction', {'image_url': viseme_image_url})
        socketio.sleep(0.1)


@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(send_predictions)


if __name__ == '__main__':
    socketio.run(app)
