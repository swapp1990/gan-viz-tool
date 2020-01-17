# import tensorflow as tf
# print("tf version {}".format(tf.__version__))
# assert tf.test.is_gpu_available()
# assert tf.test.is_built_with_cuda()

import os
import math
import time
import datetime
import sys
import logging
import statistics
import numpy as np
from matplotlib import pyplot as plt
import pickle
import PIL.Image
from PIL import ImageFilter
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
import align_images

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
from keras.applications.resnet50 import preprocess_input

alignImgDir = "server\\temp\\aligned_images\\"
encodeImgDir = "server\\temp\\encoded_images\\"
latentRepsDir = "server\\temp\\latent_reps\\"
generated_images_dir = "generated_images/"
decay_steps = 4
iterations = 100
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

        # ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
        # self.ref_images = list(filter(os.path.isfile, ref_images))
        # if len(ref_images) == 0:
        #     raise Exception('%s is empty' % src_dir)
        
        self.styleGanGenerator = None
        self.styleGanDiscriminator = None
        self.encodeGen = None
        self.perceptual_model = None
        self.dlatentGenerator = None

        self.curr_dlatents = None
        self.loss_history = []
        self.lr_history = []
        self.s1 = self.s2 = None

    def normalInit(self):
        self.initApp()
        # self.makeModel()
        # self.startTraining()
        # self.playLatent(weights=[0.3,0.7])
    
    def makeModels(self):
        self.broadcast({"log": "Making Models", "type": "replace", "logid": "makeModel"})
        self.encodeGen, self.styleGanGenerator, self.styleGanDiscriminator = self.getPretrainedStyleGanNetworks()
        self.perceptual_model = self.getPretrainedVGGPerceptualModel()
        self.dlatentGenerator = self.getPretrainedResnetModel()
        self.broadcast({"log": "Made Models", "type": "replace", "logid": "makeModel"})
        self.build()
    
    def initApp(self):
        #First get all the temp raw images and send it to client to display
        rawImgDir = "server\\temp\\raw_images"
        self.sendImgDirToClient(rawImgDir, tag='raw_images')   

    def sendImgDirToClient(self, img_dir, tag='raw_images'):
        images_list = [x for x in os.listdir(img_dir)]
        image_size = 256
        self.broadcast({'action': 'resetReceivedImgs', 'tag':tag})
        for img_name in images_list:
            img_path = os.path.join(img_dir, img_name)
            img = PIL.Image.open(img_path).convert('RGB')
            if image_size is not None:
                img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
            img = np.array(img)
            self.broadcastImg(img, tag=tag, filename=img_name)

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
            ff_model = load_model(pt_resnet_local_path)
        print("downloaded pt resnet model")
        return ff_model

    def build(self):
        self.perceptual_model.build_perceptual_model(self.encodeGen, self.styleGanDiscriminator)
        print("built perc model to train")
    
    def playLatent(self, weights=[0.3,0.7]):
        if(self.s1 == None):
            s1 = np.load('latent_rep/photo1.npy')
            s2 = np.load('latent_rep/photo2.npy')
            s1 = np.expand_dims(s1,axis=0)
            s2 = np.expand_dims(s2,axis=0)
        mixedLatent = weights[0]*s1 + weights[1]*s2
        img_array = self.generate_raw_image(mixedLatent)
        log = "Latents Weights (%f, %f)" % (weights[0], weights[1])
        self.broadcast({"log": log, "type": "replace", "logid": "pL"})
        gen_img = saveGeneratedImages(img_array, 'LatentMix', img_size=512)
        self.broadcastTrainingImages(gen_img)
    
    def generate_raw_image(self, latent_vector):
        print(latent_vector.shape)
        model_res = 1024
        model_scale = int(2*(math.log(model_res,2)-1))
        latent_vector = latent_vector.reshape((1, model_scale, 512))
        self.encodeGen.set_dlatents(latent_vector)
        return self.encodeGen.generate_images()[0]
    
    def alignImage(self, filename):
        align_images.alignImageandSave(filename)
        self.sendImgDirToClient(alignImgDir, tag='align_images')

    def encodeImage(self, filename):
        if self.encodeGen is None:
            self.makeModels()
        self.beginEncoding(filename)
        
    #################################### TRAINING ##############################
    def beginEncoding(self, filename, iterations=100):
        latentPath = latentRepsDir + filename[:-4] + '.npy'
        print("beginEncoding ", latentPath)
        if os.path.exists(latentPath):
            print('Already Encoded')
            self.broadcast({"log": "Loaded from Saved"})
            loadedLatent = np.load(latentPath)
            self.encodeGen.set_dlatents(loadedLatent)
            self.generateImgFromLatent(filename, inter_dlatent = loadedLatent)
            return
        #init losses
        losses_graphs = self.getUpdatedLossGraphs()
        self.resetLossHistory()
        msg = {'action': "gotGraph", 'val': True}
        self.broadcast(msg)

        filePath = alignImgDir + filename
        images_paths_batch = [filePath]
        self.perceptual_model.set_reference_images(images_paths_batch)
        resnet_input = preprocess_input(load_images(images_paths_batch,image_size=resnet_image_size))
        self.curr_dlatents = self.dlatentGenerator.predict(resnet_input)
        if self.curr_dlatents is not None:
            self.encodeGen.set_dlatents(self.curr_dlatents)
        self.broadcast({"log": "Generated Resnet Encoding", "type": "replace", "logid": "startEncoding"})
        self.generateImgFromLatent(filename, inter_dlatent = self.curr_dlatents)

        op = self.perceptual_model.optimize(self.encodeGen.dlatent_variable, iterations=iterations, use_optimizer='ggt')
        pbar = tqdm(op, leave=False, total=iterations)
        best_loss = None
        best_dlatent = None
        iterCount = 0
        for loss_dict in pbar:
            iterCount = iterCount + 1
            if best_loss is None or loss_dict["loss"] < best_loss:
                self.calculateLossHistory(loss_dict["loss"], loss_dict["lr"])
                losses_graphs = self.getUpdatedLossGraphs()
                self.broadcastLossHistoryFig(losses_graphs)

                if best_dlatent is None:
                    best_dlatent = self.encodeGen.get_dlatents()
                else:
                    best_dlatent = 0.25 * best_dlatent + 0.75 * self.encodeGen.get_dlatents()
                self.encodeGen.set_dlatents(best_dlatent)
                best_loss = loss_dict["loss"]
                log = "Perpetual Loss Training (%d/%d)" % (iterCount, iterations)
                self.broadcast({"log": log, "type": "replace", "logid": "perpetualLoss"})
                self.generateImgFromLatent(filename, inter_dlatent = best_dlatent)
        self.encodeGen.stochastic_clip_dlatents()
        self.encodeGen.set_dlatents(best_dlatent)
        self.generateImgFromLatent(filename, inter_dlatent = best_dlatent[0])
        self.saveLatentVector(filename, best_dlatent)
        log = "Perpetual Loss Training Finished!"
        self.broadcast({"log": log, "type": "replace", "logid": "perpetualLoss"})

    def saveLatentVector(self, filename, dlatent):
        filename = filename[:-4]
        print(filename)
        np.save(os.path.join(latentRepsDir, f'{filename}.npy'), dlatent)

    def generateImgFromLatent(self, filename, inter_dlatent=None):
        img_size = 256
        generated_images = self.encodeGen.generate_images()
        img = PIL.Image.fromarray(generated_images[0], 'RGB')
        img = img.resize((img_size,img_size),PIL.Image.LANCZOS)
        self.broadcastImg(img, tag="encoded_images", filename=filename)
        
    def resetLossHistory(self):
        self.loss_history = []
        self.lr_history = []

    def startTraining(self):
        self.broadcast({"log": "Started Training", "type": "replace", "logid": "startTraining"})
        msg = {'action': "gotGraph", 'val': True}
        self.broadcast(msg)
        self.trainPerceptualLoss()
        self.broadcast({"log": "Finished Training", "type": "replace", "logid": "startTraining"})
        msg = {'action': 'initForPlay', 'val': True}
        self.broadcast(msg)

    def trainPerceptualLoss(self):
        # self.loss_history.append(65.4)
        # self.loss_history.append(34.4)
        # losses_graphs = [{'history': self.loss_history}]
        losses_graphs = self.getUpdatedLossGraphs()
        # self.broadcastLossHistoryFig(losses_graphs)
        for images_batch in tqdm(split_to_batches(self.ref_images, batch_size), total=len(self.ref_images)//batch_size):
            self.resetLossHistory()
            names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
            self.perceptual_model.set_reference_images(images_batch)
            self.curr_dlatents = self.dlatentGenerator.predict(preprocess_input(load_images(images_batch,image_size=resnet_image_size)))
            if self.curr_dlatents is not None:
                self.encodeGen.set_dlatents(self.curr_dlatents)
            op = self.perceptual_model.optimize(self.encodeGen.dlatent_variable, iterations=iterations, use_optimizer='ggt')
            pbar = tqdm(op, leave=False, total=iterations)
            best_loss = None
            best_dlatent = None
            for loss_dict in pbar:
                pbar.set_description("Image: " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
                self.calculateLossHistory(loss_dict["loss"], loss_dict["lr"])
                losses_graphs = self.getUpdatedLossGraphs()
                self.broadcastLossHistoryFig(losses_graphs)
                if best_loss is None or loss_dict["loss"] < best_loss:
                    if best_dlatent is None:
                        best_dlatent = self.encodeGen.get_dlatents()
                    else:
                        best_dlatent = 0.25 * best_dlatent + 0.75 * self.encodeGen.get_dlatents()
                    self.encodeGen.set_dlatents(best_dlatent)
                    generated_images = self.encodeGen.generate_images()
                    train_img = saveGeneratedImages(generated_images[0])
                    self.broadcastTrainingImages(train_img)
                    # self.broadcastLossHistoryFig(losses_graphs)
                    best_loss = loss_dict["loss"]
                self.loss_history.append(loss_dict["loss"])
            self.encodeGen.stochastic_clip_dlatents()
            self.encodeGen.set_dlatents(best_dlatent)
            generated_images = self.encodeGen.generate_images()
            # print(generated_images.shape)
            generated_dlatents = self.encodeGen.get_dlatents()
            for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch, names):
                # print(img_array.shape)
                img = PIL.Image.fromarray(img_array, 'RGB')
                img.save(os.path.join("generated_images/", f'{img_name}.png'), 'PNG')
                np.save(os.path.join("latent_rep/", f'{img_name}.npy'), dlatent)
            self.encodeGen.reset_dlatents()

    def getUpdatedLossGraphs(self):
        return [{'history': self.loss_history, 'name': "Loss History"}, {'history': self.lr_history, 'name': "LR History"}]

    def trainTest(self):
        self.loss_history.append(65.4)
        self.loss_history.append(34.4)
        self.lr_history.append(3.4)
        losses_graphs = [{'history': self.loss_history}, {'history': self.lr_history}]
        self.broadcastLossHistoryFig(losses_graphs)
        for i in range(5):
            self.loss_history.append(6*i)
            self.lr_history.append(3.4*i)
            self.broadcastLossHistoryFig(losses_graphs)
    ################### Thread Methods ###################################
    def doWork(self, msg):
        print("do work StyleGanEncoding", msg)
        if msg['action'] == 'initApp':
            self.initApp()
        elif msg['action'] == 'makeModel':
            self.makeModel()
        elif msg['action'] == 'alignImage':
            self.alignImage(msg['filename'])
        elif msg['action'] == 'encodeImage':
            self.encodeImage(msg['filename'])
        elif msg['action'] == 'startTraining':
            self.startTraining()
        elif msg['action'] == 'playLatents':
            weights = [float(msg['w0']), 1-float(msg['w0'])]
            self.playLatent(weights=weights)

    def broadcast(self, msg):
        msg["id"] = 1
        workerCls.broadcast_event(msg)

    def broadcastImg(self, img, imgSize=256, tag='type', filename='filename'):
        my_dpi = 96
        img_size = (256,256)
        fig = plt.figure(figsize=(imgSize/my_dpi, imgSize/my_dpi), dpi=my_dpi)
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(img, cmap='plasma')
        # plt.show()
        mp_fig = mpld3.fig_to_dict(fig)
        plt.close('all')
        msg = {'action': 'sendImg', 'fig': mp_fig, 'tag': tag, 'filename': filename}
        self.broadcast(msg)

    def broadcastTrainingImages(self, generatedImg):
        my_dpi = 96
        img_size = (256,256)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(generatedImg, cmap='plasma')
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
    
    def broadcastLossHistoryFig(self, losses, stepGap=1):
        n_graphs = len(losses)
        fig = plt.figure(figsize=(4*n_graphs,4))
        for g_i in range(n_graphs):
            # print(losses[g_i])
            hist = losses[g_i]['history']
            label = losses[g_i]['name']
            xdata = [i*stepGap for i in range(len(hist))]
            ax1 = fig.add_subplot(1,n_graphs,g_i+1)
            ax1.plot(xdata, hist, 'c--', lw=3, label=label)
            ax1.legend()
        # plt.show()
        mp_fig = mpld3.fig_to_html(fig)
        plt.close('all')
        msg = {'action': 'sendGraph', 'fig': mp_fig}
        self.broadcast(msg)

    def calculateLossHistory(self, curr_loss, curr_lr):
        # curr_loss = curr_loss.numpy()
        l = float("{0:.2f}".format(curr_loss))
        self.loss_history.append(l)
        l2 = float("{0:.2f}".format(curr_lr))
        self.lr_history.append(l2)
        # print("calculateLossHistory ", self.lr_history)
        # currMean = statistics.mean(self.gen_lh)
        # self.mean_gen_lh.append(currMean)

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def saveGeneratedImages(img_array, filename='train_img', img_size=256):
    generated_images_dir = "generated_images/"
    img = PIL.Image.fromarray(img_array, 'RGB')
    img = img.resize((img_size,img_size),PIL.Image.LANCZOS)
    img.save(os.path.join(generated_images_dir, f'{filename}.png'), 'PNG')
    return img
################################# Socket #############################################
threadG = None
@socketio.on('init')
def init(content):
    print('init')
    workerCls.clear()

@socketio.on('alignImages')
def alignImages(filename):
    print(filename)
    msg = {'id': 0, 'action': 'alignImage', 'filename': filename}
    workerCls.broadcast_event(msg)

@socketio.on('encodeImages')
def alignImages(filename):
    print(filename)
    msg = {'id': 0, 'action': 'encodeImage', 'filename': filename}
    workerCls.broadcast_event(msg)

@socketio.on('beginTraining')
def beginTraining():
    print('beginTraining')
    stylegan_encode = StyleGanEncoding()
    threadG = workerCls.Worker(0, stylegan_encode, socketio=socketio)
    threadG.start()
    thread2 = workerCls.Worker(1, socketio=socketio)
    thread2.start()

    msg = {'id': 0, 'action': 'initApp'}
    workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'makeModel'}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'prepareData'}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'startTraining'}
    # workerCls.broadcast_event(msg)

@socketio.on('playWithLatents')
def playWithLatents(w0):
    msg = {'id': 0, 'action': 'playLatents', 'w0': w0}
    workerCls.broadcast_event(msg)

def testInit():
    styleEnc = StyleGanEncoding()
    styleEnc.normalInit()

if __name__ == "__main__":
    print("running socketio")
    # testInit()
    socketio.run(app)