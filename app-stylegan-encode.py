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
import pickle
from tqdm import tqdm
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import mpld3
from skimage.transform import resize
import seaborn as sns; sns.set()

#my classes
from server.threads import Worker as workerCls

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

################################# Style GAN Encoding #############################################
#imports
import dnnlib
import dnnlib.tflib as tflib
from keras.models import load_model
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel, load_images

src_dir = "aligned_images/"
generated_images_dir = "generated_images/"
decay_steps = 4
iterations = 500
pt_stylegan_model_url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
pt_vgg_perceptual_model_url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'
cache_dir = 'cache/'
pt_resnet_local_path = cache_dir + 'finetuned_resnet.h5'
batch_size = 1
clipping_threshold = 2.0
tile_dlatents = False
randomize_noise = False
model_res = 1024
resnet_image_size = 256

class StyleGanEncoding():
    def __init__(self):
        global decay_steps
        decay_steps *= 0.01 * iterations # Calculate steps as a percent of total iterations
        print(decay_steps)

        ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
        ref_images = list(filter(os.path.isfile, ref_images))
        if len(ref_images) == 0:
            raise Exception('%s is empty' % src_dir)
        
        self.styleGanGenerator = None
        self.styleGanDiscriminator = None
        self.encodeGen = None
        self.perceptual_model = None
        self.dlatentGenerator = None

        self.curr_dlatents = None

    def normalInit(self):
        self.encodeGen, self.styleGanGenerator, self.styleGanDiscriminator = self.getPretrainedStyleGanNetworks()
        self.perceptual_model = self.getPretrainedVGGPerceptualModel()
        self.dlatentGenerator = self.getPretrainedResnetModel()

        self.build()
    #Use StyleGAN repo's dnnlib to download the stylegan model from url or cache if already downloaded
    def getPretrainedStyleGanNetworks(self):
        tflib.init_tf()
        with dnnlib.util.open_url(pt_stylegan_model_url, cache_dir=cache_dir) as f:
            generator_network, discriminator_network, Gs_network = pickle.load(f)
        print("downloaded pt generator")
        encodeGen = Generator(Gs_network, batch_size, clipping_threshold=clipping_threshold, tiled_dlatent=False, model_res=model_res, randomize_noise=False)
        return encodeGen, Gs_network, discriminator_network

    def getPretrainedVGGPerceptualModel(self):
        with dnnlib.util.open_url(pt_vgg_perceptual_model_url, cache_dir=cache_dir) as f:
            perc_model =  pickle.load(f)
        print("downloaded pt vgg model")
        perceptual_model = PerceptualModel(perc_model=perc_model, batch_size=batch_size, decay_steps=decay_steps)
        return perceptual_model
    
    def getPretrainedResnetModel(self):
        if os.path.exists(pt_resnet_local_path):
            from keras.applications.resnet50 import preprocess_input
            ff_model = load_model(pt_resnet_local_path)
        print("downloaded pt resnet model")
        return ff_model

    def build(self):
        self.perceptual_model.build_perceptual_model(self.encodeGen, self.styleGanDiscriminator)
        print("built perc model to train")
    
    # def train():

    ################### Thread Methods ###################################
    def doWork(self, msg):
        print("do work StyleGanEncoding", msg)
        # if msg['action'] == 'makeModel':
        #     self.makeModel()
        # elif msg['action'] == 'prepareData':
        #     self.prepareData()
        # elif msg['action'] == 'startTraining':
        #     self.startTraining()
    def broadcast(self, msg):
        msg["id"] = 1
        workerCls.broadcast_event(msg)

    def broadcastTrainingImages(self, generated, target):
        # print(generated.shape, target.shape)
        generated = tf.clip_by_value(generated, clip_value_min=0, clip_value_max=1)
        generated = tf.image.resize(generated, [128,128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        target = tf.clip_by_value(target, clip_value_min=0, clip_value_max=1)
        target = tf.image.resize(target, [128,128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        genImgTile = generated[0]
        targetImgTile = target[0]
        for i in range(1,8):
            genImgTile = np.concatenate((genImgTile, generated[i]), axis=1)
        for i in range(1,8):
            targetImgTile = np.concatenate((targetImgTile, target[i]), axis=1)
        finalTile = np.concatenate((targetImgTile, genImgTile), axis=0)
        my_dpi = 96
        img_size = (128*8,128*2)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(finalTile, cmap='plasma')
        # plt.show()
        mp_fig = mpld3.fig_to_dict(fig)
        plt.close('all')
        msg = {'action': 'sendCurrTrainingFigs', 'fig': mp_fig}
        self.broadcast(msg)

    def broadcastGeneratedImages(self, const_test_noise=None):
        if const_test_noise is None:
            test_noise = np.random.randn(8, 128).astype(np.float32)
        else:
            test_noise = const_test_noise
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
        msg = {'action': 'sendRandomGeneratedFigs', 'fig': mp_fig}
        self.broadcast(msg)
        # plt.imsave('results/const.png',newImg)
    
    def broadcastLossHistoryFig(self, stepGap=1):
        xdata = [i*stepGap for i in range(len(self.gen_lh))]
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131)
        # ax1.plot(xdata, self.gen_lh, 'b-', label='gen_loss')
        ax1.plot(xdata, self.mean_gen_lh, 'c--', lw=3, label='mean_gen_loss')
        ax1.legend()

        ax2 = fig.add_subplot(132)
        # l1 = ax2.plot(xdata, self.disc_lh, 'r-', label='disc_loss')
        l2 = ax2.plot(xdata, self.mean_disc_lh, 'm--', lw=3, label='mean_disc_loss')
        l3 = ax2.plot(xdata, self.mean_real_disc_lh, 'g--', lw=3, label='mean_disc_real_loss')
        l4 = ax2.plot(xdata, self.mean_fake_disc_lh, 'y--', lw=3, label='mean_disc_fake_loss')
        ax2.legend()
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Gen Loss')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Disc Loss')

        ax3 = fig.add_subplot(133)
        l1 = ax3.plot(self.gamma_val, 'r-', label='gamma_sa')
        ax3.legend()
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
    # biggan = BigGAN()
    # threadG = workerCls.Worker(0, biggan, socketio=socketio)
    # threadG.start()
    # thread2 = workerCls.Worker(1, socketio=socketio)
    # thread2.start()

    msg = {'id': 0, 'action': 'makeModel'}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'prepareData'}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'startTraining'}
    # workerCls.broadcast_event(msg)

def testInit():
    styleEnc = StyleGanEncoding()
    styleEnc.normalInit()

if __name__ == "__main__":
    print("running socketio")

    testInit()
    socketio.run(app)