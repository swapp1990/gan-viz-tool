import tensorflow as tf
print("tf version {}".format(tf.__version__))
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

import os
import time
import datetime
import sys
import logging
import statistics
from matplotlib import pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import mpld3

#my classes
from threads import Worker as workerCls

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

################################# BIG GAN #############################################
class BigGAN():
    def __init__(self):
        self.BATCH_SIZE = 32
    
    ################### Thread Methods ###################################
    def doWork(self, msg):
        print("do work biggan", msg)
    
    def broadcast(self, msg):
        msg["id"] = 1
        workerCls.broadcast_event(msg)
################################# Socket #############################################
@socketio.on('init')
def init(content):
    print('init')

@socketio.on('beginTraining')
def beginTraining():
    # print('beginTraining')
    biggan = BigGAN()
    thread = workerCls.Worker(0, biggan, socketio=socketio)
    thread.start()
    thread2 = workerCls.Worker(1, socketio=socketio)
    thread2.start()

    msg = {'id': 0, 'action': 'makeModel'}
    workerCls.broadcast_event(msg)
#     msg = {'id': 0, 'action': 'prepareData'}
#     workerCls.broadcast_event(msg)
#     msg = {'id': 0, 'action': 'startTraining'}
#     workerCls.broadcast_event(msg)

if __name__ == "__main__":
    print("running socketio")
    socketio.run(app)