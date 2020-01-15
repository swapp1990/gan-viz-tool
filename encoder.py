import tensorflow as tf
print("tf version {}".format(tf.__version__))
import os
import math
import numpy as np
from keras.models import Model
from keras.models import load_model
import keras.backend as K
from keras.utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import traceback
import dnnlib
import dnnlib.tflib as tflib
import pickle
from functools import partial
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter

def tf_custom_adaptive_loss(a,b):
    from adaptive import lossfun
    shape = a.get_shape().as_list()
    dim = np.prod(shape[1:])
    a = tf.reshape(a, [-1, dim])
    b = tf.reshape(b, [-1, dim])
    loss, _, _ = lossfun(b-a, var_suffix='1')
    return tf.math.reduce_mean(loss)
class PerceptualModel:
    def __init__(self, batch_size=1, perc_model=None):
        self.sess = tf.get_default_session()
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.lr = 0.25
        self.decay_rate = 0.9
        self.decay_steps = 4
        self.img_size = 256
        self.layer = 9
        self.vgg_loss = 0.4
        self.discriminator_loss = 0.5
        self.discriminator = None
        self.batch_size = batch_size

        self.ref_img = None
        self.ref_weight = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None
    
    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator, discriminator=None):
        # Learning rate
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        incremented_global_step = tf.assign_add(global_step, 1)
        self._reset_global_step = tf.assign(global_step, 0)
        self.learning_rate = tf.train.exponential_decay(self.lr, incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])
        self.discriminator = discriminator
        generated_image_tensor = generator.generated_image
        # print(generated_image_tensor.shape)
        generated_image = tf.image.resize_nearest_neighbor(generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)

        self.ref_img = tf.get_variable('ref_img', shape=generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img")
        self.add_placeholder("ref_weight")
        if (self.vgg_loss is not None):
            vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
            self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
            generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight * generated_image))
            self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                        dtype='float32', initializer=tf.initializers.zeros())
            self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                                    dtype='float32', initializer=tf.initializers.zeros())
            self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
            self.add_placeholder("ref_img_features")
            self.add_placeholder("features_weight")
        
        self.loss = 0
         # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.loss += self.vgg_loss * tf_custom_adaptive_loss(self.features_weight * self.ref_img_features, self.features_weight * generated_img_features)
        print(self.loss)

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size, sharpen=True)
        image_features = None
        if self.perceptual_model is not None:
            image_features = self.perceptual_model.predict_on_batch(preprocess_input(np.array(loaded_image)))
            weight_mask = np.ones(self.features_weight.shape)
        image_mask = np.ones(self.ref_weight.shape)
        if image_features is not None:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space
            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])
            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

        images_space = list(self.ref_weight.shape[1:])
        existing_images_space = [len(images_list)] + images_space
        empty_images_space = [self.batch_size - len(images_list)] + images_space
        existing_images = np.ones(shape=existing_images_space)
        empty_images = np.zeros(shape=empty_images_space)
        image_mask = image_mask * np.vstack([existing_images, empty_images])
        loaded_image = np.vstack([loaded_image, np.zeros(empty_images_space)])

        if image_features is not None:
            self.assign_placeholder("features_weight", weight_mask)
            self.assign_placeholder("ref_img_features", image_features)
        self.assign_placeholder("ref_weight", image_mask)
        self.assign_placeholder("ref_img", loaded_image)

    def optimize(self, vars_to_optimize, iterations=100):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        fetch_ops = [min_op, self.loss, self.learning_rate]
        self.sess.run(self._reset_global_step)
        for _ in range(iterations):
            _, loss, lr = self.sess.run(fetch_ops)
            yield {"loss":loss,"lr":lr}

# Initialize generator and perceptual model
def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))
def create_variable_for_generator(name, batch_size, model_scale=18):
    return tf.get_variable('learnable_dlatents',
            shape=(batch_size, model_scale, 512),
            dtype='float32',
            initializer=tf.initializers.random_normal())
class Generator:
    def __init__(self, model, batch_size):
        self.batch_size = batch_size
        model_res = 1024
        self.model_scale = int(2*(math.log(model_res,2)-1)) # For example, 1024 -> 18
        self.initial_dlatents = np.zeros((self.batch_size, self.model_scale, 512))

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()
        model.components.synthesis.run(self.initial_dlatents,randomize_noise=False,minibatch_size=self.batch_size,custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size, model_scale=self.model_scale),
                                                        partial(create_stub, batch_size=batch_size)],structure='fixed')

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self._assign_dlatent_ph = tf.placeholder(tf.float32, name="assign_dlatent_ph")
        self._assign_dlantent = tf.assign(self.dlatent_variable, self._assign_dlatent_ph)

        def get_tensor(name):
            try:
                return self.graph.get_tensor_by_name(name)
            except KeyError:
                return None
        self.generator_output = get_tensor('G_synthesis_1/_Run/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis_1/_Run/concat/concat:0')
        if self.generator_output is None:
            self.generator_output = get_tensor('G_synthesis_1/_Run/concat_1/concat:0')
        if self.generator_output is None:
            raise Exception("Couldn't find G_synthesis_1/_Run/concat tensor output")
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)
        print(self.generated_image_uint8.shape)
    
    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def set_dlatents(self, dlatents):
        if (isinstance(dlatents.shape[0], int)):
            assert (dlatents.shape == (self.batch_size, self.model_scale, 512))
            self.sess.run([self._assign_dlantent], {self._assign_dlatent_ph: dlatents})
            return
    
    def generate_images(self, dlatents=None):
        return self.sess.run(self.generated_image_uint8)

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
def load_images(images_list, image_size=256, sharpen=False):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB')
      if image_size is not None:
        img = img.resize((image_size,image_size),PIL.Image.LANCZOS)
        if (sharpen):
            img = img.filter(ImageFilter.DETAIL)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

tflib.init_tf()
model_url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
cache_dir = '/cache'
with dnnlib.util.open_url(model_url, cache_dir=cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

generator = Generator(Gs_network, 1)
perceptual_model = PerceptualModel(batch_size=1)
perceptual_model.build_perceptual_model(generator, discriminator_network)

src_dir = "aligned_images/"
batch_size = 1
iterations = 1000
ff_model = None
load_resnet_path = "cache/finetuned_resnet.h5"
ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
ref_images = list(filter(os.path.isfile, ref_images))
#print(ref_images)
generated_images_dir = 'generated'
for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images)//batch_size):
    names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
    # print(names)
    perceptual_model.set_reference_images(images_batch)
    dlatents = None
    if (ff_model is None):
        from keras.applications.resnet50 import preprocess_input
        print("Loading ResNet Model:")
        ff_model = load_model(load_resnet_path)
    if (ff_model is not None):
        dlatents = ff_model.predict(load_images(images_batch,image_size=256))
        # print(dlatents.shape)
    if dlatents is not None:
        generator.set_dlatents(dlatents)
    op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations)
    pbar = tqdm(op, leave=False, total=iterations)
    best_loss = None
    best_dlatent = None
    for loss_dict in pbar:
        pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
        if best_loss is None or loss_dict["loss"] < best_loss:
            if best_dlatent is None:
                best_dlatent = generator.get_dlatents()
            else:
                best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()
            generator.set_dlatents(best_dlatent)
            best_loss = loss_dict["loss"]
    print(" ".join(names), " Loss {:.4f}".format(best_loss))

    #Generate images using the best latents and save them
    generated_images = generator.generate_images()
    generated_dlatents = generator.get_dlatents()
    for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch, names):
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join("generated", f'{img_name}.png'), 'PNG')
        np.save(os.path.join("generated", f'{img_name}.npy'), dlatent)