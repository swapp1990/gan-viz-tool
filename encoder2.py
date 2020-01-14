import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
import os

# load the StyleGAN model into Colab
URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
cache_dir = '/cache'
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

# load the latents
s1 = np.load('generated/stock_photo1_01.npy')
s2 = np.load('generated/stock_photo2_01.npy')
s1 = np.expand_dims(s1,axis=0)
s2 = np.expand_dims(s2,axis=0)
# combine the latents somehow... let's try an average:
savg = 0.5*(s1+s2)
# run the generator network to render the latents:
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)
im = PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB')
im.save(os.path.join("generated", f'test.png'), 'PNG')