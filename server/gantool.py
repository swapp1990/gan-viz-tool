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
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import mpld3
from skimage.transform import resize

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
        self.EPOCHS = 100
        self.DATASET_TYPE = 'celeba' #['cifar10', 'celeba']
        
        self.disc_optimizer = Adam(0.0002, beta_1=0.0, beta_2=0.9)
        self.gen_optimizer = Adam(0.0002, beta_1=0.0, beta_2=0.9)

        self.generator = None
        self.discriminator = None
        self.train_data = None

        self.gen_lh = []
        self.mean_gen_lh = []
        self.disc_lh = []
        self.mean_disc_lh = []
        self.real_dlh = []
        self.mean_real_disc_lh = []
        self.fake_dlh = []
        self.mean_fake_disc_lh = []

    def normalInit(self):
        self.makeModel()
        self.prepareData()
    
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
        sendLogMsg = "Optimizer: Adam; Loss: Wasserstein"
        print(sendLogMsg)
        self.broadcast({"log": sendLogMsg})
    
    #Returns keras.Model
    def make_generator(self):
        inp = Input(shape=(128,))
        x = Dense(4*4*512, kernel_initializer='glorot_uniform')(inp)
        x = Reshape((4,4,512))(x)
        x = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(x)
        gen_model = Model(inp, x, name='model_generator')
        return gen_model
    
    def make_discriminator(self):
        inp = Input(shape=(32,32,3))
        channels = 64
        x = Conv2D(channels, kernel_size=4, strides=2, kernel_initializer='glorot_uniform', padding='same')(inp)
        x = LeakyReLU(0.1)(x)
        channels = channels*2
        x = Conv2D(channels, kernel_size=4, strides=2, kernel_initializer='glorot_uniform', padding='same')(x)
        x = LeakyReLU(0.1)(x)
        channels = channels*2
        x = Conv2D(channels, kernel_size=4, strides=2, kernel_initializer='glorot_uniform', padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        disc_model = Model(inp, x, name='model_discriminator')
        return disc_model

    #################### Prepare Data ##################################
    def scale_dataset(self, images, new_shape):
        images_list = list()
        for img in images:
            new_img = resize(img, new_shape, 0)
            # plt.imshow(new_img)
            # plt.show()
            images_list.append(new_img)
        return np.asarray(images_list)

    def load_real_samples(self, filename):
        data = np.load(filename)
        X = data['arr_0']
        X = X.astype('float32')
        # scale from [0,255] to [-1,1]
        X = (X - 127.5) / 127.5
        return X

    def loadDataset(self, datasetName='cifar10'):
        if datasetName == 'celeba':
            DATASET_DIR = "C:\\Users\Swapinl\Documents\Datasets\\img_align_celeba\img_align_celeba_128.npz"
            X = self.load_real_samples(DATASET_DIR)
            X = self.scale_dataset(X, [32,32])
        elif datasetName=='cifar10':
            (x_train, y_train),(x_test, y_test) = cifar10.load_data()
            X = np.concatenate((x_test,x_train))
            X = X.astype('float32')
            X = X/255
        return X

    def prepareData(self):
        self.broadcast({"log": "Preparing Data", "type": "replace", "logid": "loaddata"})
        self.train_data = self.loadDataset(datasetName=self.DATASET_TYPE)
        sendLogMsg = "Prepared Data: Datatype: %s, Total Images %d, of size (%dx%d)"%(self.DATASET_TYPE, self.train_data.shape[0], self.train_data.shape[1], self.train_data.shape[2])
        print(sendLogMsg)
        self.broadcast({"log": sendLogMsg, "type": "replace", "logid": "loaddata"})
    
    ################### Start Training ###################################
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true*y_pred)

    def train_step(self, noise_inputs, target, epoch, n):
        gen_model = self.generator
        disc_model = self.discriminator
        real_y = np.ones((32, 1), dtype=np.float32)
        fake_y = -real_y
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_type:
            gen_output = gen_model(noise_inputs, training=True)
            disc_real_output = disc_model(target, training=True)
            disc_fake_output = disc_model(gen_output, training=True)
            gen_loss = self.wasserstein_loss(disc_fake_output, real_y)
            disc_real_loss = self.wasserstein_loss(disc_real_output, real_y)
            disc_fake_loss = self.wasserstein_loss(disc_fake_output, fake_y)
            disc_loss = disc_real_loss + disc_fake_loss
        disc_gradients = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, disc_model.trainable_variables))
        gen_gradients = gen_type.gradient(gen_loss, gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, gen_model.trainable_variables))
        if n%20 == 0:
            self.broadcastGeneratedImages()
            tf.print(gen_loss)
            tf.print(disc_real_loss, disc_fake_loss, disc_loss)
            self.calculateLossHistory(gen_loss, disc_loss, disc_real_loss, disc_fake_loss)
            self.broadcastLossHistoryFig()

    def fit(self, train_ds, epochs):
        for e in range(epochs):
            sendLogMsg = "Training Epoch %d/%d"%(e, epochs)
            self.broadcast({"log": sendLogMsg, "type": "replace", "logid": "epoch"})
            bs = self.BATCH_SIZE
            np.random.shuffle(train_ds)
            num_batches = int(train_ds.shape[0] // bs)
            # index_total = np.empty((0), int)
            for n in range(num_batches):
                disc_minibatches = train_ds[n*bs:(n+1)*bs]
                noise_inputs = np.random.randn(bs, 128).astype(np.float32)
                self.train_step(noise_inputs, disc_minibatches, e, n)
                if n%50 == 0:
                    sendLogMsg = "Step %d/%d"%(n, num_batches)
                    self.broadcast({"log": sendLogMsg, "type": "replace", "logid": "step"})

    def startTraining(self):
        self.fit(self.train_data, self.EPOCHS)

    ################### Thread Methods ###################################
    def doWork(self, msg):
        print("do work biggan", msg)
        if msg['action'] == 'makeModel':
            self.makeModel()
        elif msg['action'] == 'prepareData':
            self.prepareData()
        elif msg['action'] == 'startTraining':
            self.startTraining()
    def broadcast(self, msg):
        msg["id"] = 1
        workerCls.broadcast_event(msg)

    def broadcastGeneratedImages(self):
        test_noise = np.random.randn(8, 128).astype(np.float32)
        gen_imgs = self.generator.predict(test_noise)
        gen_imgs = tf.clip_by_value(gen_imgs, clip_value_min=0, clip_value_max=1)
        gen_imgs = tf.image.resize(gen_imgs, [128,128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        newImg = gen_imgs[0]
        for i in range(1,8):
            newImg = np.concatenate((newImg, gen_imgs[i]), axis=1)
        my_dpi = 96
        img_size = (128*8,128)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(1,1,1)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(title[i])
        ax.imshow(newImg, cmap='plasma')
        mp_fig = mpld3.fig_to_dict(fig)
        plt.close('all')
        msg = {'action': 'sendFigs', 'fig': mp_fig}
        self.broadcast(msg)
        # plt.imsave('results/const.png',newImg)
    
    def broadcastLossHistoryFig(self, stepGap=1):
        xdata = [i*stepGap for i in range(len(self.gen_lh))]
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(121)
        ax1.plot(xdata, self.gen_lh, 'b-', label='gen_loss')
        ax1.plot(xdata, self.mean_gen_lh, 'c--', lw=3, label='mean_gen_loss')
        ax1.legend()

        ax2 = fig.add_subplot(122)
        l1 = ax2.plot(xdata, self.disc_lh, 'r-', label='disc_loss')
        l2 = ax2.plot(xdata, self.mean_disc_lh, 'm--', lw=3, label='mean_disc_loss')
        l3 = ax2.plot(xdata, self.mean_real_disc_lh, 'g--', lw=3, label='mean_disc_real_loss')
        l4 = ax2.plot(xdata, self.mean_fake_disc_lh, 'y--', lw=3, label='mean_disc_fake_loss')
        ax2.legend()
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Gen Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Disc Loss')
        # plt.show()
        mp_fig = mpld3.fig_to_html(fig)
        plt.close('all')
        msg = {'action': 'sendGraph', 'fig': mp_fig}
        self.broadcast(msg)

    def calculateLossHistory(self, gen_loss, disc_loss, disc_rl, disc_fl):
        gen_loss = gen_loss.numpy()
        l = float("{0:.2f}".format(gen_loss))
        self.gen_lh.append(l)
        currMean = statistics.mean(self.gen_lh)
        self.mean_gen_lh.append(currMean)

        disc_loss = disc_loss.numpy()
        l2 = float("{0:.2f}".format(disc_loss))
        self.disc_lh.append(l2)
        self.mean_disc_lh.append(statistics.mean(self.disc_lh))

        disc_rl = disc_rl.numpy()
        disc_rl = float("{0:.2f}".format(disc_rl))
        self.real_dlh.append(disc_rl)
        self.mean_real_disc_lh.append(statistics.mean(self.real_dlh))
        disc_fl = disc_fl.numpy()
        disc_fl = float("{0:.2f}".format(disc_fl))
        self.fake_dlh.append(disc_fl)
        self.mean_fake_disc_lh.append(statistics.mean(self.fake_dlh))
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
    msg = {'id': 0, 'action': 'prepareData'}
    workerCls.broadcast_event(msg)
    msg = {'id': 0, 'action': 'startTraining'}
    workerCls.broadcast_event(msg)

def testInit():
    biggan = BigGAN()
    biggan.normalInit()

if __name__ == "__main__":
    print("running socketio")
    socketio.run(app)

    # testInit()