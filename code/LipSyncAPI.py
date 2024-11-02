from flask import Flask, render_template
from flask_socketio import SocketIO

from lipsync_pytorch import run_lipsync


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('lipsync_anim.html')


def send_predictions():
    for viseme in run_lipsync():
        socketio.emit('new_prediction', {'viseme': viseme})
        socketio.sleep(0.1)


@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(send_predictions)


if __name__ == '__main__':
    socketio.run(app)
