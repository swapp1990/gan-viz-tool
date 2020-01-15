import os
import pickle
import config
import dnnlib
import gzip
import json
import numpy as np
from tqdm import tqdm_notebook
import warnings
import matplotlib.pylab as plt
warnings.filterwarnings("ignore")
import dnnlib.tflib as tflib
import PIL.Image
from encoder.generator_model import Generator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

#Load latent face labels for random vars
LATENT_TRAINING_DATA = 'https://drive.google.com/uc?id=1xMM3AFq0r014IIhBLiMCjKJJvbhLUQ9t'
stylegan_model_url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
cache_dir = "cache/"
    
with dnnlib.util.open_url(LATENT_TRAINING_DATA, cache_dir=cache_dir) as f:
    qlatent_data, dlatent_data, labels_data = pickle.load(gzip.GzipFile(fileobj=f))

print(len(dlatent_data), dlatent_data[0].shape, len(labels_data))
tflib.init_tf()
with dnnlib.util.open_url(stylegan_model_url, cache_dir=cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
# images = Gs_network.components.synthesis.run(dlatent_data[0], minibatch_size=1, randomize_noise=False, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), structure='fixed')
# print(images.shape)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1,18, 512))
    generator.set_dlatents(latent_vector)
    img_arr = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_arr, 'RGB')
    return img.resize((256,256))

def move_and_show(latent_vector, direction, coeffs):
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        ax[i].imshow(generate_image(new_latent_vector))
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.show()

# Let's play with age and gender
X_data = dlatent_data.reshape((-1, 18*512))
# print(X_data.shape) #(20307, 9216)
y_age_data = np.array([x['faceAttributes']['age'] for x in labels_data])
y_gender_data = np.array([x['faceAttributes']['gender'] == 'male' for x in labels_data])
assert(len(X_data) == len(y_age_data) == len(y_gender_data))
print(len(X_data))

clf = LogisticRegression(class_weight='balanced')
clf.fit(X_data.reshape((-1, 18*512)), y_gender_data)
gender_direction = clf.coef_.reshape((18, 512))

move_and_show(X_data.reshape((-1, 18, 512))[1], gender_direction, [-5, -1.5, 0, 1.5, 5])

#Distribution
# plt.hist(y_age_data[y_gender_data], bins=30, color='red',alpha=0.5,label='male')
# plt.hist(y_age_data[~y_gender_data],bins=30, color='blue',alpha=0.5,label='female')
# plt.legend()
# plt.title('Distribution of age within gender')
# plt.xlabel('Age')
# plt.ylabel('Population')
# plt.show()

#Train a linear model for obtaining gender direction in latent space


# clf = LogisticRegression(class_weight='balanced').fit(X_data, y_gender_data)
# gender_direction = clf.coef_.reshape((18, 512))
# print(gender_direction.shape)

# clf = SGDClassifier('log', class_weight='balanced') # SGB model for performance sake
# scores = cross_val_score(clf, X_data, y_gender_data, scoring='accuracy', cv=5)
# clf.fit(X_data, y_gender_data)
# print(scores)
# print('Mean: ', np.mean(scores))

# bins, bin_edges = np.histogram(y_age_data, bins=30)
# errors,_ = np.histogram(y_age_data[clf.predict(X_data) != y_gender_data], bin_edges)

# plt.plot(errors / bins)
# plt.title('Dependency of gender detection errors on age')
# plt.ylabel('Gender detection error rate')
# plt.xlabel('Age')
# plt.show()

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

model.fit(X_data.reshape((-1, 18*512)), y_gender_data, validation_split=0.2, epochs=5)
model = Model(model.input, model.layers[-2].output)
