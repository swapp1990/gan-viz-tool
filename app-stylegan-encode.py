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
import tensorflow as tf
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
import gzip
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from scipy.io import wavfile
import moviepy.editor

alignImgDir = "server\\temp\\aligned_images\\"
encodeImgDir = "server\\temp\\encoded_images\\"
latentRepsDir = "server\\temp\\latent_reps\\"
latentAttrDir = "server\\temp\\latent_attr\\"
generated_images_dir = "generated_images/"
decay_steps = 4
iterations = 100
# pt_stylegan_model_url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
pt_stylegan_model_url = 'https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3' #Cars
pt_vgg_perceptual_model_url = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2'
LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
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
        self.attrDirVec = None

        self.curr_dlatents = None
        self.loss_history = []
        self.lr_history = []
        self.s1 = self.s2 = None

        #style mixing
        self.src_dlatents_w_seeds = []
        self.dst_dlatents_w_seeds = []

    def normalInit(self):
        # self.initApp()
        # self.makeModels()
        # self.playLatent(weights=[0.3,0.7])
        # self.loadAttributes()
        # self.drawFigures()
        # self.loadStyleMixing()
        # self.performStyleMixing()
        # self.loadNoiseMixer(config={'minLayer': '0', 'maxLayer': 15})
        # self.testMusicEncoding()
        self.ganSteer()
    
    ##################################### Gan Steer ##############################################
    def ganSteer(self):
        tflib.init_tf()
        with dnnlib.util.open_url(pt_stylegan_model_url, cache_dir=cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        Nsliders = 3 # we use 3 slider dimensions for RGB color

        dim_z = Gs.input_shape[1]

        # get original generated output
        z = tf.placeholder(tf.float32, shape=(None, dim_z))
        outputs_orig = tf.transpose(Gs.get_output_for(z, None, is_validation=True, 
                                                    randomize_noise=True), [0, 2, 3, 1])

        img_size = outputs_orig.shape[1]
        Nchannels = outputs_orig.shape[3]

        # set target placeholders
        target = tf.placeholder(tf.float32, shape=(None, img_size, img_size, Nchannels))
        mask = tf.placeholder(tf.float32, shape=(None, img_size, img_size, Nchannels))

        # forward to W latent space
        out_dlatents = Gs.components.mapping.get_output_for(z, None) #out_dlatents shape: [?, 16, 512]

        # set slider and learnable walk vector
        latent_dim = out_dlatents.shape
        alpha = tf.placeholder(tf.float32, shape=(None, Nsliders), name="alpha_slider")
        w = tf.Variable(np.random.normal(0.0, 0.1, [1, latent_dim[1], latent_dim[2], Nsliders]), name='walk_intermed', dtype=np.float32)

        # apply walk
        out_dlatents_new = out_dlatents
        for i in range(Nsliders):
            out_dlatents_new = out_dlatents_new + tf.reshape(
                tf.expand_dims(alpha[:,i], axis=1)* tf.reshape(w[:,:,:,i], (1, -1)), (-1, 16, z.shape[1]))

        # get output after applying walk
        transformed_output = tf.transpose(Gs.components.synthesis.get_output_for(
            out_dlatents_new, is_validation=True, randomize_noise=True), [0, 2, 3, 1])

        # L_2 loss
        loss = tf.losses.compute_weighted_loss(tf.square(transformed_output-target), weights=mask)

        # ops to rescale the stylegan output range ([-1, 1]) to uint8 range [0, 255]
        float_im = tf.placeholder(tf.float32, outputs_orig.shape)
        uint8_im = tflib.convert_images_to_uint8(tf.convert_to_tensor(float_im, dtype=tf.float32))

        def initialize_uninitialized(sess):
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            print([str(i.name) for i in not_initialized_vars])
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
                return not_initialized_vars
        
        lr = 0.001
        num_samples = 2000
        sess = tf.get_default_session()
        not_initialized_vars = initialize_uninitialized(sess)

        # change to loss_lpips to optimize using lpips loss instead
        train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=not_initialized_vars, 
                                                        name='AdamOpter')

        # this time init Adam's vars:
        not_initialized_vars = initialize_uninitialized(sess)

        def testInterSteer(Nsliders, z_inputs):
            a = np.linspace(0, 1, 6) #[0. 0.2 0.4 0.6 0.8 1.]
            # print("z_inputs ", z_inputs.shape)
            # ims = []
            for i in range(a.shape[0]):
                alpha_val_test = np.ones((1, Nsliders)) * -a[i]
                #Suppress Red and Blue (using slider), but increase one color (G)
                alpha_val_test[:, 1] = a[i]
                target_fn,_ = get_target_np(out_zs, alpha_val_test)

            #     im_out = sess.run(transformed_output, {z: zs[s], alpha: alpha_val_test})
            #     #Scale
            #     im_out = sess.run(uint8_im, {float_im: im_out})
            #     target_fn = sess.run(uint8_im, {float_im: target_fn})
            #     ims.append(im_out)
            # canvas = PIL.Image.new('RGB', (512*6, 512), 'white')
            # for j, img in enumerate(ims):
            #     canvas.paste(PIL.Image.fromarray(img[0], 'RGB'), (512*j, 0))
            # canvas.save('trained.png')

        def get_target_np(outputs_zs, alpha):
            if not np.any(alpha): # alpha is all zeros
                return outputs_zs, np.ones(outputs_zs.shape)
            
            assert(outputs_zs.shape[0] == alpha.shape[0])
            
            target_fn = np.copy(outputs_zs)
            for b in range(outputs_zs.shape[0]):
                for i in range(3):
                    target_fn[b,:,:,i] = target_fn[b,:,:,i]+alpha[b,i]

            mask_out = np.ones(outputs_zs.shape)
            return target_fn, mask_out
        
        saver = tf.train.Saver(tf.trainable_variables(scope='walk'))
        loss_vals = []
        random_seed = 0
        rnd = np.random.RandomState(random_seed)
        zs = rnd.randn(num_samples, dim_z)

        n_epoch = 10
        optim_iter = 0
        batch_size = 4
        for epoch in range(n_epoch):
            for batch_start in range(0, num_samples, batch_size):
                alpha_val = np.random.random(size=(batch_size, Nsliders))-0.5
                s = slice(batch_start, min(num_samples, batch_start + batch_size))

                #get original images (by passing batch of random z-values)
                #the original image is generated by the pretrained stylegan model
                feed_dict = {z: zs[s]}
                out_orig_imgs = sess.run(outputs_orig, feed_dict=feed_dict)
                
                #Apply random alpha vals to change the slider (rgb) values in the image randomly and get the target image
                target_imgs, mask_out = get_target_np(out_orig_imgs, alpha_val)

                #Feed the target image to the optimizer, which "walks" or changes the intermediate w-mapping. This slightly different w-mapping generates similar image, but forces it to have features similar to target image. In this example, target image has slightly different rgb values applied.
                #The loss used is l2_loss, which simply calculates pixel-wise differences in the generated and target image.
                feed_dict = {z: zs[s], alpha: alpha_val, target: target_imgs, mask: mask_out}
                curr_loss, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                print(batch_start, curr_loss)

                #For testing
                #generated_imgs - get the transformed output images from the sess graph, with the current random alpha_val applied to the current mini-batch of random z-values as input
                feed_dict = {z: zs[s], alpha: alpha_val}
                    #you must supply all tf.placeholder values to get intermediate values from the sess graph
                generated_imgs = sess.run(transformed_output, feed_dict=feed_dict)

                #Rescale the img from ([-1, 1]) to uint8 range [0, 255]
                out_orig_imgs = sess.run(uint8_im, {float_im: out_orig_imgs})
                generated_imgs = sess.run(uint8_im, {float_im: generated_imgs})
                target_imgs = sess.run(uint8_im, {float_im: target_imgs})
                #Display canvas with the transformed img
                canvas = PIL.Image.new('RGB', (512*3, 512), 'white')
                canvas.paste(PIL.Image.fromarray(out_orig_imgs[0], 'RGB'))
                canvas.paste(PIL.Image.fromarray(generated_imgs[0], 'RGB'), (512*1,0))
                canvas.paste(PIL.Image.fromarray(target_imgs[0], 'RGB'), (512*2,0))
                canvas.save('trained.png')

                #Test a single seed image while training to see the changes in the color
                a = np.linspace(0, 1, 6)
                sample_seed = 0
                s = slice(sample_seed, sample_seed + 1)

                # print(zs[s].shape, zs[sample_seed].shape)
                feed_dict = {z: zs[s]}
                orig_imgs = sess.run(outputs_orig, feed_dict=feed_dict)
                canvas = PIL.Image.new('RGB', (512*a.shape[0], 512), 'white')
                for i in range(a.shape[0]):
                    alpha_val_test = np.ones((zs[s].shape[0], Nsliders)) * -a[i]
                    alpha_val_test[:, 1] = a[i]
                    target_img,_ = get_target_np(orig_imgs, alpha_val_test)
                    im_out = sess.run(transformed_output, {z: zs[s], alpha: alpha_val_test})
                    im_out = sess.run(uint8_im, {float_im: im_out})
                    canvas.paste(PIL.Image.fromarray(im_out[0], 'RGB'), (512*i,0))
                canvas.save('trained2.png')
        saver.save(sess, '{}/model_{}.ckpt'.format('output', optim_iter*batch_size), write_meta_graph=False, write_state=False)

    def saveOrShowSamplesFromModel(self, Gs, showPlt=False, filename='samples.png'):
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
        img_size = 512 #Should be the output size of model using 'get_output_for'
        src_seeds=[0,1,2,3,4]
        src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        src_dlatents = Gs.components.mapping.run(src_latents, None)
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
        canvas = PIL.Image.new('RGB', (img_size * len(src_seeds), img_size), 'white')
        for col, src_image in enumerate(list(src_images)):
            canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (col * img_size, 0))
        if showPlt:
            plt.imshow(canvas)
            plt.grid(False)
            plt.axis('off')
            plt.show()
        else:
            canvas.save(filename)
    ##################################### Gan Steer ##############################################

    def makeModels(self, loadPerpetual=False):
        self.broadcast({"log": "Making Models", "type": "replace", "logid": "makeModel"})
        self.encodeGen, self.styleGanGenerator, self.styleGanDiscriminator = self.getPretrainedStyleGanNetworks()
        if loadPerpetual:
            self.perceptual_model = self.getPretrainedVGGPerceptualModel()
            self.dlatentGenerator = self.getPretrainedResnetModel()
            self.build()
        self.broadcast({"log": "Made Models", "type": "replace", "logid": "makeModel"})
    
    def initApp(self, config):
        print(config)
        if(config['alreadyEncoded']):
            self.makeModels(loadPerpetual=False)
            self.sendSavedEncodingsToClient()
        else:
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

    def sendSavedEncodingsToClient(self):
        encoding_list = [x for x in os.listdir(latentRepsDir)]
        print("%d saved encodings found" % (len(encoding_list)))
        for enc_path in encoding_list:
            self.sendEncodingFromFile(enc_path)

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
    
    def playLatent(self, weights=[0.3,0.7], images=[]):
        latentPath1 = latentRepsDir + images[0][:-4] + '.npy'
        latentPath2 = latentRepsDir + images[1][:-4] + '.npy'
        if os.path.exists(latentPath1):
            s1 = np.load(latentPath1)
            s2 = np.load(latentPath2)
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
        # align_images.alignImageandSave(filename)
        self.sendImgDirToClient(alignImgDir, tag='align_images')

    def encodeImage(self, filename):
        if self.encodeGen is None:
            isLoadPerc = self.checkIfEncodingExists(filename)
            isLoadPerc = not isLoadPerc
            self.makeModels(loadPerpetual=isLoadPerc)
        self.beginEncoding(filename)
    
    def checkIfEncodingExists(self, filename):
        latentPath = latentRepsDir + filename[:-4] + '.npy'
        print("beginEncoding ", latentPath)
        if os.path.exists(latentPath):
            print('Already Encoded')
            return True
        return False
        
    #################################### TRAINING ##############################
    def sendEncodingFromFile(self, filename):
        latentPath = latentRepsDir + filename
        if os.path.exists(latentPath):
            loadedLatent = np.load(latentPath)
            self.encodeGen.set_dlatents(loadedLatent)
            img = self.generateImgFromLatent(filename, inter_dlatent = loadedLatent)
            self.broadcastImg(img, tag="encoded_images", filename=filename)

    def beginEncoding(self, filename, iterations=100):
        latentPath = latentRepsDir + filename[:-4] + '.npy'
        print("beginEncoding ", latentPath)
        if os.path.exists(latentPath):
            print('Already Encoded')
            self.broadcast({"log": "Loaded from Saved"})
            loadedLatent = np.load(latentPath)
            self.encodeGen.set_dlatents(loadedLatent)
            img = self.generateImgFromLatent(filename, inter_dlatent = loadedLatent)
            self.broadcastImg(img, tag="encoded_images", filename=filename)
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

        img = self.generateImgFromLatent(filename, inter_dlatent = self.curr_dlatents)
        self.broadcastImg(img, tag="encoded_images", filename=filename)

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
                img = self.generateImgFromLatent(filename, inter_dlatent = best_dlatent)
                self.broadcastImg(img, tag="encoded_images", filename=filename)

        self.encodeGen.stochastic_clip_dlatents()
        self.encodeGen.set_dlatents(best_dlatent)

        img = self.generateImgFromLatent(filename, inter_dlatent = best_dlatent[0])
        self.broadcastImg(img, tag="encoded_images", filename=filename)

        self.saveLatentVector(filename, best_dlatent)
        log = "Perpetual Loss Training Finished!"
        self.broadcast({"log": log, "type": "replace", "logid": "perpetualLoss"})

    def saveLatentVector(self, filename, dlatent):
        filename = filename[:-4]
        np.save(os.path.join(latentRepsDir, f'{filename}.npy'), dlatent)

    def generateImgFromLatent(self, filename, inter_dlatent=None):
        img_size = 256
        generated_images = self.encodeGen.generate_images()
        img = PIL.Image.fromarray(generated_images[0], 'RGB')
        img = img.resize((img_size,img_size),PIL.Image.LANCZOS)
        return img
        
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
    
    ##################### Get attribute latent directions ######################
    def loadAttributes(self, config=None):
        print('loadAttributes ', config)
        
        with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=cache_dir) as f:
            qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))
        if config is None:
            print(labels_data[0])
            return
        X_data = dlatent_data.reshape((-1, 18*512))
        # Let's play with config type ['age' or 'gender']
        # print(X_data.shape) #(20307, 9216)
        
        attr_type = config['type']
        attr_factor = config['factor']
        if attr_type == 'age':
            if attr_factor == '<15':
                attr_factor = int(15)
            elif attr_factor == '15-25':
                attr_factor = int(20)
            elif attr_factor == '25-35':
                attr_factor = int(30)
        if attr_type == 'hair':
            attr_factor = 'bald'
        attr_dirn_filename = attr_type + "_" + str(attr_factor)
        attr_latent_path = latentAttrDir + attr_dirn_filename + '.npy'
        if os.path.exists(attr_latent_path):
            self.attrDirVec = np.load(attr_latent_path)
            logMsg = "Loaded %s Direction Vector from cache" % (attr_type)
            self.broadcast({"log": logMsg})
        else:
            if attr_type != 'hair' and attr_type != 'emotion':
                y_data = np.array([x['faceAttributes'][attr_type] == attr_factor for x in labels_data])
            elif attr_type == 'hair':
                y_data = np.array([x['faceAttributes'][attr_type]['bald'] > 0.8 for x in labels_data])
            elif attr_type == 'emotion':
                y_data = np.array([x['faceAttributes'][attr_type][attr_factor] > 0.8 for x in labels_data])
            assert(len(X_data) == len(y_data))
            logMsg = "Performing regression on %s Direction on Sample Data" % (attr_type)
            self.broadcast({"log": logMsg})
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(X_data.reshape((-1, 18*512)), y_data)
            self.broadcast({"log": "Regression Done successfully and found the direction vector"})
            self.attrDirVec = clf.coef_.reshape((18, 512))
            np.save(os.path.join(latentAttrDir, f'{attr_dirn_filename}.npy'), self.attrDirVec)
    
    def generateImg_withAttrDirVec(self, filename, coeff=-1.5):
        #Generate orig image using the 'filename'.npy latent vector file
        latentPath1 = latentRepsDir + filename[:-4] + '.npy'
        img_size = 512
        if os.path.exists(latentPath1):
            origLatent = np.load(latentPath1)
            #Move the latent towards the direction of the attribute using the 'coeff' value
            new_latent_vector = origLatent.copy()
            new_latent_vector[:8] = (origLatent + coeff*self.attrDirVec)[:8]
            new_latent_vector = new_latent_vector.reshape((1,18, 512))
            self.encodeGen.set_dlatents(new_latent_vector)
            img_arr = self.encodeGen.generate_images()[0]
            img = PIL.Image.fromarray(img_arr, 'RGB')
            img = img.resize((img_size,img_size),PIL.Image.LANCZOS)
            # gen_img = saveGeneratedImages(img_array, 'LatentMix', img_size=512)
            self.broadcastTrainingImages(img)
            # plt.imshow(img)
            # plt.show()
   
    def drawFigures(self):
        tflib.init_tf()
        with dnnlib.util.open_url(pt_stylegan_model_url, cache_dir=cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        
        w=1024 
        h=1024
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
        #Uncurated results
        # lods=[0,1,2,2,3,3]
        # rows = 3
        # cols = 3
        # seed=5
        # latents = np.random.RandomState(seed).randn(sum(rows * 2**lod for lod in lods), Gs.input_shape[1])
        # images = Gs.run(latents, None, **synthesis_kwargs) # [seed, y, x, rgb]
        # self.showImagesAsGrid(images)

        #Style Mixing
        # src_seeds=[639,701,687,615,2268] 
        # dst_seeds=[888,829,1898,1733,1614,845]
        # style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)]
        # src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        # dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
        # src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
        # dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
        # src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
        # dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

        # canvas = PIL.Image.new('RGB', (1024 * (len(src_seeds) + 1), 1024 * (len(dst_seeds) + 1)), 'white')
        # for col, src_image in enumerate(list(src_images)):
        #     canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * 1024, 0))
        # for row, dst_image in enumerate(list(dst_images)):
        #     canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * 1024))
        #     row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        #     row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        #     row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        #     for col, image in enumerate(list(row_images)):
        #         canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * 1024, (row + 1) * 1024))
        # plt.imshow(canvas)
        # plt.grid(False)
        # plt.axis('off')
        # plt.show()

        #Noise detail

        # seeds=[1157,1012]
        # num_samples=100
        # canvas = PIL.Image.new('RGB', (w * 3, h * len(seeds)), 'white')
        # for row, seed in enumerate(seeds):
        #     latents = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples)
        #     images = Gs.run(latents, None, truncation_psi=1, **synthesis_kwargs)
        #     canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (0, row * h))
        #     for i in range(4):
        #         crop = PIL.Image.fromarray(images[i + 1], 'RGB')
        #         crop = crop.crop((650, 180, 906, 436))
        #         crop = crop.resize((w//2, h//2), PIL.Image.NEAREST)
        #         canvas.paste(crop, (w + (i%2) * w//2, row * h + (i//2) * h//2))
        #     diff = np.std(np.mean(images, axis=3), axis=0) * 4
        #     diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        #     canvas.paste(PIL.Image.fromarray(diff, 'L'), (w * 2, row * h))
        # plt.imshow(canvas)
        # plt.grid(False)
        # plt.axis('off')
        # plt.show()

        #Noise components
        seeds=[1967,1555]
        flips=[1]
        noise_ranges=[range(0, 18), range(0, 0), range(8, 18), range(0, 8)]
        Gsc = Gs.clone()
        noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('noise')]
        noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]
        latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
        all_images = []
        for noise_range in noise_ranges:
            tflib.set_vars({var: val * (1 if i in noise_range else 0) for i, (var, val) in enumerate(noise_pairs)})
            range_images = Gsc.run(latents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
            range_images[flips, :, :] = range_images[flips, :, ::-1]
            all_images.append(list(range_images))
        
        canvas = PIL.Image.new('RGB', (w * 2, h * 2), 'white')
        for col, col_images in enumerate(zip(*all_images)):
            canvas.paste(PIL.Image.fromarray(col_images[0], 'RGB').crop((0, 0, w//2, h)), (col * w, 0))
            canvas.paste(PIL.Image.fromarray(col_images[1], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, 0))
            canvas.paste(PIL.Image.fromarray(col_images[2], 'RGB').crop((0, 0, w//2, h)), (col * w, h))
            canvas.paste(PIL.Image.fromarray(col_images[3], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, h))
        canvas.save("noiseComp.png")

    def showImagesAsGrid(self, images):
        canvas = PIL.Image.new('RGB', (256 * 3, 256 * 3), 'white')
        image_iter = iter(list(images))
        for i in range(0,256*3,256):
            for j in range(0,256*3,256):
                image = PIL.Image.fromarray(next(image_iter), 'RGB')
                # image = image.crop((cx, cy, cx + cw, cy + ch))
                image = image.resize((256, 256), PIL.Image.ANTIALIAS)
                canvas.paste(image, (i, j))
        plt.imshow(canvas)
        plt.grid(False)
        plt.axis('off')
        plt.show()
    
    ##################### Style Mixing ######################
    def loadStyleMixing(self, config=None):
        if self.encodeGen is None:
            self.makeModels()
            Gs = self.styleGanGenerator
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
        src_seeds=[639,701,687,615,2268] 
        dst_seeds=[888,829,1898,1733,1614]
        # src_seeds = np.random.randint(low=1, high=3000, size=(5)).tolist()
        # dst_seeds = np.random.randint(low=1, high=3000, size=(5)).tolist()
        print(src_seeds, dst_seeds)
        src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
        dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
        src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
        dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
        dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, print_progress=True, **synthesis_kwargs)

        image_size=256
        for seed,img in zip(src_seeds, src_images):
            img = PIL.Image.fromarray(img, 'RGB')
            img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
            self.broadcastImg(img, tag="src_images", filename=seed)
        for seed,img in zip(dst_seeds, dst_images):
            img = PIL.Image.fromarray(img, 'RGB')
            img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
            self.broadcastImg(img, tag="dest_images", filename=seed)
        
        self.src_dlatents_w_seeds = [{'seed':a, 'dlatent':b} for a,b in zip(src_seeds, src_dlatents)]
        self.dst_dlatents_w_seeds = [{'seed':a, 'dlatent':b} for a,b in zip(dst_seeds, dst_dlatents)]

    def performStyleMixing(self, config=None):
        if self.styleGanGenerator is not None:
            Gs = self.styleGanGenerator
        print('performStyleMixing ', config)
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
        if config is None:
            minL = 0
            maxL = 0
            src_seed = 639
            dst_seed = 888
        else:
            minL = int(config['minLayer'])
            maxL = int(config['maxLayer'])
            src_seed = int(config['src'])
            dst_seed = int(config['dest'])
        
        layers_to_mix = range(minL,maxL)
        src_dlatent_dict = next((item for item in self.src_dlatents_w_seeds if item["seed"] == src_seed), None)
        if src_dlatent_dict is not None:
            src_dlatent = src_dlatent_dict['dlatent']
        dst_dlatent_dict = next((item for item in self.dst_dlatents_w_seeds if item["seed"] == dst_seed), None)
        if dst_dlatent_dict is not None:
            dst_dlatent = dst_dlatent_dict['dlatent']
        # print(dst_dlatent.shape, src_dlatent.shape)

        #Mix the src and dst dlatents according to the layers specified
        mixed_dlatent = dst_dlatent.copy()
        mixed_dlatent[layers_to_mix] = src_dlatent[layers_to_mix]
        #Expand dlatent shape [18,512] => [1,18,512] to match Gs required shape
        mixed_dlatent = np.stack([mixed_dlatent * 1])
        
        row_images = Gs.components.synthesis.run(mixed_dlatent, randomize_noise=False, **synthesis_kwargs)
        img = PIL.Image.fromarray(row_images[0], 'RGB')
        image_size = 512
        img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
        self.broadcastTrainingImages(img)
        # plt.imshow(img)
        # plt.show()
    
    def loadNoiseMixer(self, config=None):
        print("loadNoiseMixer ", config)
        seed = 1967
        if self.styleGanGenerator is None:
            self.makeModels()
        Gs = self.styleGanGenerator

        # for name, var in Gs.components.synthesis.vars.items():
        #     if name.startswith('noise'):
        #         print(var.shape)
        for name, var in Gs.components.synthesis.vars.items():
            print(name, var.shape)
        # synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
        # Gsc = Gs.clone()
        # noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('noise')]
        # noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]
        print(Gs.components.synthesis.output_templates)
        # conv_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('512x512/Conv0_up/weight')]
        # conv_pairs = list(zip(conv_vars, tflib.run(conv_vars)))
        # seeds = [seed]
        # zlatents = np.stack(np.random.RandomState(seed).randn(Gsc.input_shape[1]) for seed in seeds)
        # minL = int(config['minLayer'])
        # maxL = int(config['maxLayer'])
        # layers_with_noise = range(minL,maxL)
        #4 long curly, #6 mid curly, #8 short curly 10 wavy+curly hair, 12 - dotted
        layers_with_noise = [6,8,10]
        # noise_range = range(0, 8)
        image_size = 256

        ######################### Var dict way ####################
        # var_dict = {}
        # for i, (var, val) in enumerate(noise_pairs):
        #     if i not in layers_with_noise:
        #         scale = 0
        #         val = val * scale
        #     else:
        #         scale = 1
        #         # if i == 8:
        #         #     scale = 0.2
        #         val = val * scale
        #     var_dict[var] = val
        # tflib.set_vars(var_dict)
        # range_images = Gsc.run(zlatents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
        # img = PIL.Image.fromarray(range_images[0], 'RGB')
        # img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
        #########################
        all_images = []
        allPrevNoise = []
        # for j in range(0,18):
        #     dict_var = {var: val * (1 if i == j else 0) for i, (var, val) in enumerate(noise_pairs)}
        #     allPrevNoise.append(dict_var)
        #     tflib.set_vars(dict_var)
        #     range_images = Gsc.run(zlatents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
        #     img = PIL.Image.fromarray(range_images[0], 'RGB')
        #     img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
        #     all_images.append(img)
        # self.saveImgWithAllNoise(all_images, filename="baby.png")
        # self.broadcastTrainingImages(img)

        # self.saveImgWithAllNoise(all_images, filename="baby.png")
        # self.saveImgWithAllNoise(all_images, filename="caren.png")

    def saveImgWithAllNoise(self, all_images, image_size=256, filename="const.png"):
        canvas = PIL.Image.new('RGB', (image_size*18, image_size), 'white')
        for i in range(18):
            canvas.paste(all_images[i], (i*image_size,0))
        canvas.save(filename)
    
    ##################### Music Encoding ######################
    def testMusicEncoding(self):
        audio = {}
        fps = 60
        for mp3_filename in [f for f in os.listdir('music') if f.endswith('.mp3')]:
            mp3_filename = f'music/{mp3_filename}'
            wav_filename = mp3_filename[:-4] + '.wav'
            if not os.path.exists(wav_filename):
                audio_clip = moviepy.editor.AudioFileClip(mp3_filename)
                audio_clip.write_audiofile(wav_filename, fps=44100, nbytes=2, codec='pcm_s16le')
            track_name = os.path.basename(wav_filename)[15:-5]
            print(os.path.basename(wav_filename))
            rate, signal = wavfile.read(wav_filename)
            print(rate, signal.shape)
            signal = np.mean(signal, axis=1) # to mono (2 channels to 1)
            signal = np.abs(signal)
            seed = signal.shape[0]
            duration = signal.shape[0] / rate
            frames = int(np.ceil(duration * fps))
            samples_per_frame = signal.shape[0] / frames
            audio[track_name] = np.zeros(frames, dtype=signal.dtype)
            print(audio)
            for frame in range(frames):
                start = int(round(frame * samples_per_frame))
                stop = int(round((frame + 1) * samples_per_frame))
                audio[track_name][frame] = np.mean(signal[start:stop], axis=0)
            audio[track_name] /= max(audio[track_name])
        print(audio.keys())
        # for track in sorted(audio.keys()):
        #     plt.figure(figsize=(8, 3))
        #     plt.title(track)
        #     plt.plot(audio[track])
        #     plt.show()

        if self.styleGanGenerator is None:
            self.makeModels()
        Gs = self.styleGanGenerator
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        Gs_syn_kwargs = dnnlib.EasyDict()
        Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_syn_kwargs.randomize_noise = False
        Gs_syn_kwargs.minibatch_size = 4
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        w_avg = Gs.get_var('dlatent_avg')

        def normalize_vector(v):
            return v * np.std(w_avg) / np.std(v) + np.mean(w_avg) - np.mean(v)

        def get_ws(n, frames, seed):
            filename = f'cache/ws_{n}_{frames}_{seed}.npy'
            if not os.path.exists(filename):
                src_ws = np.random.RandomState(seed).randn(n, 512)
                ws = np.empty((frames, 512))
                for i in range(512):
                    x = np.linspace(0, 3*frames, 3*len(src_ws), endpoint=False)
                    y = np.tile(src_ws[:, i], 3)
                    x_ = np.linspace(0, 3*frames, 3*frames, endpoint=False)
                    y_ = interp1d(x, y, kind='quadratic', fill_value='extrapolate')(x_)
                    ws[:, i] = y_[frames:2*frames]
                np.save(filename, ws)
            else:
                ws = np.load(filename)
            return ws

        def render_frame(t):
            global base_index
            frame = np.clip(np.int(np.round(t * fps)), 0, frames - 1)
            base_index += base_speed * audio['Instrumental'][frame]**2
            base_w = base_ws[int(round(base_index)) % len(base_ws)]
            base_w = np.tile(base_w, (18, 1))
            psi = 0.5 + audio['FX'][frame] / 2
            base_w = w_avg + (base_w - w_avg) * psi
            mix_w = np.tile(mix_ws[frame], (18, 1))
            mix_w = w_avg + (mix_w - w_avg) * 0.75
            ranges = [range(0, 4), range(4, 8), range(8, 18)]
            values = [audio[track][frame] for track in ['Drums', 'E Drums', 'Synth']]
            w = mix_styles(base_w, mix_w, zip(ranges, values))
            w += mouth_open * audio['Vocal'][frame] * 1.5
            image = Gs.components.synthesis.run(np.stack([w]), **Gs_syn_kwargs)[0]
            image = PIL.Image.fromarray(image).resize((size, size), PIL.Image.LANCZOS)
            return np.array(image)

        size = 1080
        seconds = int(np.ceil(duration))
        resolution = 10
        base_frames = resolution * frames
        base_ws = get_ws(seconds, base_frames, seed)
        base_speed = base_frames / sum(audio['']**2)
        base_index = 0
        mix_ws = get_ws(seconds, frames, seed + 1)
        mouth_open = normalize_vector(-np.load('cache/mouth_ratio.npy'))

        mp4_filename = 'cache/CultureShock.mp4'
        video_clip = moviepy.editor.VideoClip(render_frame, duration=duration)
        audio_clip_i = moviepy.editor.AudioFileClip('data/Culture Shock (Instrumental).wav')
        audio_clip_v = moviepy.editor.AudioFileClip('data/Culture Shock (Vocal).wav')
        audio_clip = moviepy.editor.CompositeAudioClip([audio_clip_i, audio_clip_v])
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(mp4_filename, fps=fps, codec='libx264', audio_codec='aac', bitrate='8M')
    ################### Thread Methods ###################################
    def doWork(self, msg):
        # print("do work StyleGanEncoding", msg)
        if msg['action'] == 'initApp':
            self.initApp(msg['config'])
        elif msg['action'] == 'makeModel':
            self.makeModel()
        elif msg['action'] == 'alignImage':
            self.alignImage(msg['filename'])
        elif msg['action'] == 'encodeImage':
            self.encodeImage(msg['filename'])
        elif msg['action'] == 'startTraining':
            self.startTraining()
        elif msg['action'] == 'playLatents':
            # print(msg['input'])
            w0 = msg['input']['latentWs']
            weights = [float(w0), 1-float(w0)]
            self.playLatent(weights=weights, images= msg['input']['images'])
        elif msg['action'] == 'loadAttributes':
            self.loadAttributes(msg['config'])
        elif msg['action'] == 'generateImgWithDir':
            coeff = float(msg['coeff'])
            self.generateImg_withAttrDirVec(msg['filename'], coeff=coeff)
        elif msg['action'] == 'loadStyleMixing':
            self.loadStyleMixing(msg['config'])
        elif msg['action'] == 'performStyleMixing':
            self.performStyleMixing(msg['config'])
        elif msg['action'] == 'loadNoiseMixer':
            self.loadNoiseMixer(msg['config'])

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

def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1,18, 512))
    generator.set_dlatents(latent_vector)
    img_arr = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_arr, 'RGB')
    return img.resize((256,256))

def move_and_show(latent_vector, direction, coeffs, gen):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(new_latent_vector, gen))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.show()

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

@socketio.on('initApp')
def initApp(config):
    print('initApp')
    stylegan_encode = StyleGanEncoding()
    threadG = workerCls.Worker(0, stylegan_encode, socketio=socketio)
    threadG.start()
    thread2 = workerCls.Worker(1, socketio=socketio)
    thread2.start()

    # msg = {'id': 0, 'action': 'initApp', 'config': config}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'makeModel'}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'prepareData'}
    # workerCls.broadcast_event(msg)
    # msg = {'id': 0, 'action': 'startTraining'}
    # workerCls.broadcast_event(msg)

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

@socketio.on('playWithLatents')
def playWithLatents(selectedInp):
    msg = {'id': 0, 'action': 'playLatents', 'input': selectedInp}
    workerCls.broadcast_event(msg)

@socketio.on('loadAttributes')
def loadAttributes(msg):
    msg = {'id': 0, 'action': 'loadAttributes', 'config': msg}
    workerCls.broadcast_event(msg)

@socketio.on('loadStyleMixing')
def loadAttributes(msg):
    msg = {'id': 0, 'action': 'loadStyleMixing', 'config': msg}
    workerCls.broadcast_event(msg)

@socketio.on('loadNoiseMixer')
def loadAttributes(msg):
    msg = {'id': 0, 'action': 'loadNoiseMixer', 'config': msg}
    workerCls.broadcast_event(msg)

@socketio.on('performStyleMixing')
def loadAttributes(msg):
    msg = {'id': 0, 'action': 'performStyleMixing', 'config': msg}
    workerCls.broadcast_event(msg)

@socketio.on('generateImgWithDir')
def generateImgWithDir(msg):
    # print(msg['images'])
    msg = {'id': 0, 'action': 'generateImgWithDir', 'filename': msg['images'][0], 'coeff': msg['coeff']}
    workerCls.broadcast_event(msg)

def testInit():
    styleEnc = StyleGanEncoding()
    styleEnc.normalInit()

if __name__ == "__main__":
    print("running socketio")
    testInit()
    # socketio.run(app)