"""
@Project : AcouWrite
@File : server.py
@Date : 2022/12/4 22:58
@Author : Qiuyang Zeng
@Software : PyCharm
https://blog.csdn.net/yannanxiu/article/details/52916892
"""
import os
import time
import torch
import datetime
import numpy as np
from net import Net
from cachelib import SimpleCache  # 缓存 跨请求
from transceiver.receiver import Receiver
from flask import Flask, request, session

from utils.plot_utils import show_signals
from word_suggestion.word_suggestion import spell_correction
cache = SimpleCache()

UPLOAD_FOLDER = r"../audio"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'acoustic_write'
app.permanent_session_lifetime = datetime.timedelta(seconds=10*60)
# app.config['net'] = get_net()


def get_net(path=r'model/hidden_size_16.pth'):
    args = ["M", 32, "M", 64, "M", 64, "GAP"]
    net = Net(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
              num_classes=26).cuda()
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)
    net.eval()  # 禁用 dropout, 避免 BatchNormalization 重新计算均值和方差
    return net


net = get_net(path=r'../model/params_50epochs.pth')
# net = get_digit_net(path=r'../model/digits_20epochs.pth')
# net = get_stroke_net(path=r'../model/strokes_40epochs.pth')


@app.route('/')
def hello_world():
    return 'Welcome to Acouwrite!'


@app.route('/recognition/file', methods=['POST'])
def recognition_file():
    audio = request.files.get("audio", None)
    audio.save(os.path.join(app.config['UPLOAD_FOLDER'], audio.filename))
    split_d_cir = Receiver.receive_real_time(base_path=app.config['UPLOAD_FOLDER'],
                                             filename=audio.filename)
    output_letter = ""
    for d_cir in split_d_cir:
        d_cir = torch.tensor(d_cir).float() / 255
        d_cir = d_cir.unsqueeze(0)  # add batch_size dim: torch.Size([1, 4, 1, 60, 40])
        output = net(d_cir.cuda(), [d_cir.shape[1]])
        predicted = torch.argmax(output, 0)
        output_letter = output_letter + chr(predicted.cpu().numpy() + ord('a'))
        # output_letter = output_letter + chr(predicted.cpu().numpy() + ord('0'))
    # output_letter = spell_correction(output_letter)
    print("result: {}".format(output_letter))
    return output_letter + " " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


@app.route('/recognition/start', methods=['GET'])
def start_session():
    session['audios'] = np.array([])
    return 'ok'


@app.route('/recognition/stop', methods=['GET'])
def stop_session():
    session.clear()
    return 'ok'


@app.route('/recognition/stream', methods=['POST'])
def recognition_stream():
    stream_bytes = request.files.get("stream", None).read()
    audio_bytes = np.reshape(np.frombuffer(stream_bytes, dtype=np.int16), (-1, 2), order='F')[:, 1]
    # print(session['data'])cache.set('a', '1')

    show_signals(audio_bytes)
    # audio.save(os.path.join(app.config['UPLOAD_FOLDER'], audio.filename))
    return cache.get('a')


if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0', port=5000, debug=True)
