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
from keras.datasets import cifar100, cifar10
from keras.layers.pooling import _GlobalPooling2D
from tensorflow.keras.layers import Input, Dense, Conv2D, Add, Dot, Conv2DTranspose, Activation, Reshape, LeakyReLU, Flatten, BatchNormalization, InputSpec, UpSampling2D, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from keras.utils.generic_utils import Progbar
import tensorflow.keras.backend as K
class BigGAN():
    def __init__(self):
        self.BATCH_SIZE = 32

        self.generator = None
        self.discriminator = None
    def normalInit(self):
        self.makeModel()
    
    #################### Create Model ##################################
    def makeModel(self):
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        sendLogMsg = "Made Models: Generator and Discriminator"
        print(sendLogMsg)
        self.broadcast({"log": sendLogMsg})
        modelsJson = [self.generator.to_json(), self.discriminator.to_json()]
        msg = {'action': 'sendModelsJson', 'modelsJson': modelsJson}
        print(msg['action'])
        self.broadcast(msg)
    
    #Returns keras.Model
    def make_generator(self):
        inp = Input(shape=(128,))
        x = Dense(4*4*512, kernel_initializer='glorot_uniform')(inp)
        x = Reshape((4,4,512))(x)
        x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x)
        gen_model = Model(inp, x, name='model_generator')
        return gen_model
    
    def make_discriminator(self):
        inp = Input(shape=(32,32,3))
        channels = 64
        x = Conv2D(channels, kernel_size=4, strides=2, kernel_initializer='glorot_uniform', padding='same')(inp)
        x = LeakyReLU(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        disc_model = Model(inp, x, name='model_discriminator')
        return disc_model

    ################### Thread Methods ###################################
    def doWork(self, msg):
        print("do work biggan", msg)
        if msg['action'] == 'makeModel':
            self.makeModel()

    def broadcast(self, msg):
        msg["id"] = 1
        workerCls.broadcast_event(msg)
################################# Socket #############################################
threadG = None
@socketio.on('init')
def init(content):
    print('init')
    workerCls.clear()

@socketio.on('beginTraining')
def beginTraining():
    # print('beginTraining')
    biggan = BigGAN()
    threadG = workerCls.Worker(0, biggan, socketio=socketio)
    threadG.start()
    thread2 = workerCls.Worker(1, socketio=socketio)
    thread2.start()

    msg = {'id': 0, 'action': 'makeModel'}
    workerCls.broadcast_event(msg)
#     msg = {'id': 0, 'action': 'prepareData'}
#     workerCls.broadcast_event(msg)
#     msg = {'id': 0, 'action': 'startTraining'}
#     workerCls.broadcast_event(msg)

def testInit():
    biggan = BigGAN()
    biggan.normalInit()

if __name__ == "__main__":
    print("running socketio")
    socketio.run(app)

    # testInit()