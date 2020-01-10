import tensorflow as tf
import tensorflow.keras.backend as K
import os
from math import floor, log2
from random import random
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar100, cifar10
from skimage.transform import resize
import tensorflow_datasets as tfds
from PIL import Image

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import initializers, regularizers, constraints

class Conv2DMod(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', dilation_rate=1, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, demod=True, **kwargs):
        super(Conv2DMod, self).__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.demod = demod
        self.input_spec = [InputSpec(ndim = 4),
                            InputSpec(ndim = 2)]
    
    def build(self, input_shape):
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')

        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, name='kernel', regularizer=self.kernel_regularizer, constraint = self.kernel_constraint)
        # Set input spec.
        self.input_spec = [InputSpec(ndim=4, axes={channel_axis: input_dim}),
                            InputSpec(ndim=2)]
        self.built = True
    
    def call(self, inputs):
        #To channels last
        x = tf.transpose(inputs[0], [0, 3, 1, 2])
        #Get weight and bias modulations
        #Make sure w's shape is compatible with self.kernel
        w = K.expand_dims(K.expand_dims(K.expand_dims(inputs[1], axis = 1), axis = 1), axis = -1)
        #Add minibatch layer to weights
        wo = K.expand_dims(self.kernel, axis = 0)
        #Modulate
        weights = wo * (w+1)
        #Demodulate
        if self.demod:
            d = K.sqrt(K.sum(K.square(weights), axis=[1,2,3], keepdims = True) + 1e-8)
            weights = weights / d
        
        #Reshape/scale input
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        x = tf.nn.conv2d(x, w,
                strides=self.strides,
                padding="SAME",
                data_format="NCHW")

        # Reshape/scale output.
        x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
        x = tf.transpose(x, [0, 2, 3, 1])

        return x
    
    def compute_output_shape(self, input_shape):
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'demod': self.demod
        }
        base_config = super(Conv2DMod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

im_size = 128

def do_upsample(x):
    return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')

def upsample_to_size(x):
    y = im_size / x.shape[2]
    x = K.resize_images(x, y, y, "channels_last",interpolation='bilinear')
    return x

def crop_to_fit(x):
    height = x[1].shape[1]
    width = x[1].shape[2]
    return x[0][:, :height, :width, :]

def to_rgb(inp, style):
    size = inp.shape[2]
    x = Conv2DMod(3,1,kernel_initializer=VarianceScaling(200/size),demod=False)([inp,style])
    return Lambda(upsample_to_size, output_shape=[None, im_size, im_size, None])(x)

def d_block(inp, ch, downsample=True):
    skip = Conv2D(ch, 1, kernel_initializer='he_uniform')(inp)
    x = Conv2D(ch, kernel_size=3,padding='same', kernel_initializer='he_uniform')(inp)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(ch, kernel_size=3,padding='same', kernel_initializer='he_uniform')(inp)
    x = LeakyReLU(0.2)(x)

    out = add([skip, x])
    if downsample:
        out = AveragePooling2D()(out)
    return out

def g_block(inp, istyle, inoise, ch, upsample=True):
    if upsample:
        #custom upsampling because of clone_model issue
        x = Lambda(do_upsample, output_shape=[None, inp.shape[2]*2,inp.shape[2]*2, None])(inp)
    else:
        x = Activation('linear')(inp)
    
    rgb_style = Dense(ch, kernel_initializer=VarianceScaling(200/x.shape[2]))(istyle)
    style = Dense(inp.shape[-1], kernel_initializer='he_uniform')(istyle)
    delta = Lambda(crop_to_fit)([inoise, x])
    d = Dense(ch, kernel_initializer='zeros')(delta)

    x = Conv2DMod(filters=ch, kernel_size=3, padding='same', kernel_initializer = 'he_uniform')([x,style])
    x = add([x,d])
    x = LeakyReLU(0.2)(x)

    style = Dense(ch, kernel_initializer = 'he_uniform')(istyle)
    d = Dense(ch, kernel_initializer = 'zeros')(delta)
    x = Conv2DMod(filters = ch, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([x, style])
    x = add([x, d])
    x = LeakyReLU(0.2)(x)

    return x, to_rgb(x, rgb_style)

class StyleGan(object):
    def __init__(self):
        self.DATASIZE = 2048*4
        self.learning_rate = 0.0001
        self.steps = 1000
        self.beta = 0.999
        self.im_size = 128
        self.n_layers = int(log2(self.im_size) - 1)
        self.latent_dim = 512
        self.bs = 16
        self.DATASET_TYPE = 'celeba' #['cifar10', 'celeba']
        self.train_data = []
        self.stepsTaken = 0
        
        self.pl_mean = 0
        self.D = None
        self.S = None
        self.G = None
        self.GM = None

        self.disc_optim = Adam(lr = self.learning_rate, beta_1 = 0, beta_2 = 0.999)
        self.gen_optim = Adam(lr = self.learning_rate, beta_1 = 0, beta_2 = 0.999)

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
            # DATASET_DIR = "C:\\Users\Swapinl\Documents\Datasets\\img_align_celeba\img_align_celeba_128.npz"
            DATASET_DIR = "C:\\Users\Swapinl\Documents\Datasets\\img_align_celeba\\temp"
            nm_imgs = np.sort(os.listdir(DATASET_DIR))
            print(nm_imgs)
            X = None
            # X = get_npdata(nm_imgs)
            # X = self.load_real_samples(DATASET_DIR)
        elif datasetName=='cifar10':
            (x_train, y_train),(x_test, y_test) = cifar10.load_data()
            X = np.concatenate((x_test,x_train))
            X = X.astype('float32')
            X = X/255
            X = self.scale_dataset(X, [128,128])
        return X

    def prepareData(self):
        # self.train_data = self.loadDataset(datasetName=self.DATASET_TYPE)
        total_data = tfds.load(name='celeb_a', split='train')
        for img in total_data.take(self.DATASIZE):
            self.train_data.append(img['image'].numpy())
        self.train_data = self.scale_dataset(self.train_data, [128,128])
        # plt.imshow(self.train_data[0])
        # plt.show()
    
    def noise(self, n):
        return np.random.normal(0.0, 1.0, size=[n, self.latent_dim])

    def mixedList(self, n):
        tt = int(random() * self.n_layers)
        p1 = [self.noise(n)]*tt
        p2 = [self.noise(n)]*(self.n_layers - tt)
        return p1 + [] + p2
    
    def noiseList(self, n):
        return [self.noise(n)] * self.n_layers

    def makeModel(self):
        self.D = self.make_discriminator()
        # print(self.D.summary())
        self.G = self.make_generator()
        # print(self.G.summary())
        self.GM = self.GenModel()

        self.G_cloned = clone_model(self.G)
        self.G_cloned.set_weights(self.G.get_weights())
        self.S_cloned = clone_model(self.S)
        self.S_cloned.set_weights(self.S.get_weights())

    def make_discriminator(self):
        ch = 24
        inp = Input(shape = [self.im_size, self.im_size, 3]) #128,3
        x = d_block(inp, 1*ch) #64,24
        x = d_block(x, 2*ch) #32,48
        x = d_block(x, 4*ch) #16,96
        x = d_block(x, 6*ch) #8,144
        x = d_block(x, 8*ch) #4,192
        x = d_block(x, 16*ch, downsample=False) #4,384
        x = Flatten()(x)
        x = Dense(1, kernel_initializer='he_uniform')(x)

        disc_model = Model(inputs=inp, outputs=x)
        return disc_model
    
    def make_generator(self):
        ch = 24
        self.S = Sequential()
        self.S.add(Dense(512, input_shape=[self.latent_dim]))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))

        #Inputs
        inp_style = []
        # print(self.n_layers)
        for i in range(self.n_layers):
            inp_style.append(Input([self.latent_dim]))
        inp_noise = Input(shape = [self.im_size, self.im_size, 1]) #128,1

        #?
        x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])
        outs = []

        x = Dense(4*4*4*ch, kernel_initializer='random_normal')(x)
        x = Reshape((4,4,4*ch))(x) #4,96

        x,r = g_block(x,inp_style[0],inp_noise,32*ch,upsample=False) #4,768
        outs.append(r)
        x, r = g_block(x, inp_style[1], inp_noise, 16 * ch)  #8, 384
        outs.append(r)
        x, r = g_block(x, inp_style[2], inp_noise, 8 * ch)  #16
        outs.append(r)
        x, r = g_block(x, inp_style[3], inp_noise, 6 * ch)  #32
        outs.append(r)
        x, r = g_block(x, inp_style[4], inp_noise, 4 * ch)   #64
        outs.append(r)
        x, r = g_block(x, inp_style[5], inp_noise, 2 * ch)   #128
        last_g_block = x
        last_rgb = r
        outs.append(r)
        # print(x.shape)

        x = add(outs)
        x = Lambda(lambda y: y/2 + 0.5)(x) #Use values centered around 0, but normalize to [0, 1], providing better initialization

        gen_model = Model(inputs=inp_style + [inp_noise], outputs = [x, [last_g_block, last_rgb]])
        # for v in gen_model.trainable_variables:
        #     print(v.name)
        return gen_model

    def GenModel(self):
        inp_style = []
        style = []

        for i in range(self.n_layers):
            inp_style.append(Input([self.latent_dim]))
            style.append(self.S(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])
        gf, lgb = self.G(style + [inp_noise])
        GM_model = Model(inputs = inp_style + [inp_noise], outputs = gf)
        # print(GM_model.summary())
        return GM_model
    ################### Start Training ###################################
    def gradient_penalty(self, samples, output, weight):
        g = K.gradients(output, samples)[0]
        g_sqr = K.square(g)
        g_penalty = K.sum(g_sqr, axis=np.arange(1, len(g_sqr.shape)))
        #(w/2) * ||grad||^2
        return K.mean(g_penalty) * weight

    def parameterAverage(self):
        for i in range(len(self.G.layers)):
            beta_mix = 0.8
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.G_cloned.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * beta_mix + (1-beta_mix) * up_weight[j])
            self.G.layers[i].set_weights(new_weight)
        
        for i in range(len(self.S.layers)):
            beta_mix = 0.8
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.S_cloned.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * beta_mix + (1-beta_mix) * up_weight[j])
            self.S.layers[i].set_weights(new_weight)
        
        print("paramter averaged")
    
    def saveOldWeightsForAverage(self):
        self.G_cloned.set_weights(self.G.get_weights())
        self.S_cloned.set_weights(self.S.get_weights())
        print('saved old weights')

    @tf.function
    def train_step(self, real_images, styles, noise, perform_gp=True, perform_pl=False):
        #Use this gen model to generate images to give to Discriminator
        gen_model = self.G
        disc_model = self.D
        #Use this gen model for taking gradients loss from D to train both input styles layers and Generator layers
        training_gen_model = self.GM
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            #w_space expands the input style noise, through several dense layers (which learn as the network is trained) (what does it learn ? maybe the style information (how ?))
            
            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(styles)):
                w_space.append(self.S(styles[i]))
            # print(w_space[0].shape) #(16,512)
            #generate images
            #input: noise + styles (as increased size w_space)
            g_inp = w_space + [noise] #(16,512) + (16,128,128,1) => (16,512)
            # print("g_inp ", g_inp[0].shape)
            generated_images, [last_g_block, last_r] = gen_model(g_inp)
            # tf.print(last_r.shape)
            # print(generated_images.shape) #(16,128,128,3)
            real_output = disc_model(real_images, training=True)
            fake_output = disc_model(generated_images, training=True)
            # tf.print(type(generated_images[0]))
            # plt.imshow(generated_images[0])
            # plt.show()

            #Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            #Perform Gradient penalty
            if perform_gp:
                disc_loss += self.gradient_penalty(real_images, real_output, 10)
            
            #Perform perceptual path length penalty
            returned_pl_images = None
            if perform_pl:
                #slightly adjust W space
                w_space_2 = []
                for i in range(len(styles)):
                    std = 0.1/(K.std(w_space[i], axis = 0, keepdims = True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i]))/(std + 1e-8))
                
                #Generate from slightly adjusted W space
                pl_images,lgb = self.G(w_space_2 + [noise])

                #Get distance after adjustment (path length)
                delta_g = K.mean(K.square(pl_images - generated_images), axis = [1,2,3])
                pl_lengths = delta_g
                # tf.print("pl_lengths ", pl_lengths[0])

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))
                returned_pl_images = pl_images
        disc_gradients = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
        self.disc_optim.apply_gradients(zip(disc_gradients, disc_model.trainable_variables))
        gen_gradients = gen_tape.gradient(gen_loss, training_gen_model.trainable_variables)
        self.gen_optim.apply_gradients(zip(gen_gradients, training_gen_model.trainable_variables))

        return disc_loss, gen_loss, pl_lengths, generated_images, [returned_pl_images, last_g_block[0], last_r[0]]
    
    def train(self):
        epochs = 100
        #train alternating
        mixed_prob = 0.9
        #? Why is some prob mixes list used in the noise ?
        if random() < mixed_prob:
            styles = self.mixedList(self.bs)
        else:
            styles = self.noiseList(self.bs)
        #styles is a list of random values of 'latent_dim' 512 for each batch, size of 16 passed to each layer of generator seperately
        # print(len(styles), styles[0].shape, styles[0])
        #noise is a random noise generated as the shape of the noise image with 1 channel
        noise = np.random.uniform(0.0, 1.0, size = [self.bs, im_size, im_size, 1]).astype('float32')
        num_batches = int(self.train_data.shape[0] // self.bs)
        print(num_batches)
        for e in range(epochs):
            np.random.shuffle(self.train_data)
            print('Epochs ', e)
            for i in range(num_batches):
                real_images = self.train_data[i*self.bs:(i+1)*self.bs].astype('float32')
                apply_gp = self.stepsTaken % 2 == 0
                apply_pl = self.stepsTaken % 16 == 0
                disc_loss, gen_loss, pl_lengths, generated_images, [pl_generated_imgs, last_g_block, last_r] = self.train_step(real_images, styles, noise, perform_gp=apply_gp, perform_pl=apply_pl)
                #adjust path length penalty mean
                if self.pl_mean == 0:
                    self.pl_mean = np.mean(pl_lengths)
                self.pl_mean = 0.99*self.pl_mean + 0.01*np.mean(pl_lengths)
                if self.stepsTaken % 50 == 0:
                    self.saveOldWeightsForAverage()
                if self.stepsTaken % 200 == 0:
                    self.parameterAverage()
                g = tf.clip_by_value(generated_images, clip_value_min=0, clip_value_max=1)
                last_r = tf.clip_by_value(last_r, clip_value_min=0, clip_value_max=1)
                plt.imsave('const.png',g[0].numpy())
                plt.imsave('last_r.png',last_r.numpy())
                if pl_generated_imgs is not None:
                    pl_g = tf.clip_by_value(pl_generated_imgs, clip_value_min=0, clip_value_max=1)
                    plt.imsave('pl_img.png',g[1].numpy())
                
                # showImg = last_g_block[:,:,0]
                # plt.imshow(last_g_block[:,:,0], cmap='gray')
                # plt.show()
                self.visualizePlots(last_g_block)

                if self.stepsTaken % 100 == 0:
                    print("Round " + str(self.stepsTaken) + ":")
                    print("D:", np.array(disc_loss))
                    print("G:", np.array(gen_loss))
                    # print("P:", self.pl_mean)
                    #Save Model
                    if self.stepsTaken % 500 == 0:
                        self.save(floor(self.stepsTaken/10000))
                self.stepsTaken = self.stepsTaken + 1
    
    def visualizePlots(self, features):
        xi,yj = 4,4
        f, axarr = plt.subplots(xi,yj)
        for i in range(xi):
            for j in range(yj):
                axarr[i,j].imshow(features[:,:,i+j], cmap='gray')
                axarr[i,j].axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig('g_last_block.png')
        plt.close('all')

    def save(self, num):
        self.saveModel(self.S, "sty", num)
        self.saveModel(self.G, "gen", num)
        self.saveModel(self.D, "dis", num)
        print("model saved")
    
    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("Models/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num):
        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()
        mod = model_from_json(json, custom_objects = {'Conv2DMod': Conv2DMod})
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")
        print(mod)
        return mod
    
    def load(self, num):
        self.D = self.loadModel("dis", num)
        self.S = self.loadModel("sty", num)
        self.G = self.loadModel("gen", num)
        print("Loaded from saved")
        self.GM = self.GenModel()

    def evaluate(self, num = 0, trunc = 1.0):
        n1 = self.noiseList(64)
        n2 = np.random.uniform(0.0, 1.0, size = [64, im_size, im_size, 1]).astype('float32')
        trunc = np.ones([64, 1]) * trunc
        generated_images = self.GM.predict(n1 + [n2], batch_size = self.bs)
        r = []
        for i in range(0,64,8):
            r.append(np.concatenate(generated_images[i:i+8], axis=1))
        
        c1 = np.concatenate(r, axis=0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))
        x.save('Results/i'+str(num)+"-ema.png")

styleGan = StyleGan()
styleGan.prepareData()
styleGan.makeModel()
styleGan.load(2)
styleGan.train()

