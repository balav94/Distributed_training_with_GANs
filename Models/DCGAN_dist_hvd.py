import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time as time
import pickle
import math
from PIL import Image as im
from tensorflow.keras.callbacks import History

import horovod.tensorflow as hvd
# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print("*********** Current Rank:", hvd.rank(), "on", hvd.size())

BATCH_SIZE = int(sys.argv[1])
epochs = int(sys.argv[2])  # In practice, use ~100 epochs
IMG_SIZE = (64, 64)

file_name="../../celeba_gan/np_"+ str(hvd.size()) +"_rank_"+ str(hvd.rank()) +".pickle"

print("file_name :", file_name)

# open a file, where you stored the pickled data
file = open(file_name, 'rb')

# unpickle information from that file
dataset = pickle.load(file)

print("len(dataset) :", len(dataset))

#close the file
file.close()

print('dataset[0].shape :', dataset[0].shape)

for i in range(len(dataset)):
    data = im.fromarray(dataset[i])
    data = data.resize(IMG_SIZE)
    dataset[i] = np.array(data)
    
dataset = np.array(dataset)

dataset = dataset.reshape(dataset.shape[0], 64, 64, 3)

dataset = tf.data.Dataset.from_tensor_slices(dataset)

dataset = dataset.shuffle(10000).batch(BATCH_SIZE)

normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255.0, offset=0)

dataset = dataset.map(lambda x: normalization_layer(x))

print('dataset :', dataset)

init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64*1, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64*2, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64*4, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64*8, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        # layers.Conv2D(1, 4, 2,padding='same', kernel_initializer=init, use_bias=False, activation='sigmoid')
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(1 * 1 * 128),
        layers.Reshape((1, 1, 128)),
        layers.Conv2DTranspose(64*8, kernel_size=4, strides=4, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64*4, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64*2, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64*1, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.BatchNormalization(momentum=0.1,  epsilon=0.8),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=init, activation='tanh')
    ],
    name="generator",
)
generator.summary()

d_loss_metric = keras.metrics.Mean(name="d_loss")
g_loss_metric = keras.metrics.Mean(name="g_loss")

d_optimizer = tf.optimizers.Adam(learning_rate = 0.0002*hvd.size(), beta_1 = 0.5, beta_2 = 0.999)
g_optimizer = tf.optimizers.Adam(learning_rate = 0.0002*hvd.size(), beta_1 = 0.5, beta_2 = 0.999)
# opt = tf.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999)

loss_fn = keras.losses.BinaryCrossentropy()

@tf.function
def train_step(real_images, first_batch):
    # Sample random points in the latent space
    batch_size = tf.shape(real_images)[0]
    print("batch_size :", batch_size)
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    
    print("discriminator_1")
    
    # Train Discriminator with real labels
    with tf.GradientTape() as tape_1:
        real_output = discriminator(real_images)
        real_targets = tf.ones_like(real_output)
        # Perform Label smoothing - assign a random integer in range [0.7, 1.0] for positive class
        real_targets = tf.cast(real_targets, dtype = 'float32') - 0.3*tf.random.uniform(tf.shape(real_output), minval=0, maxval=1)
        d_loss = loss_fn(real_targets, real_output)
        
    # Horovod: add Horovod Distributed GradientTape.
    tape_1 = hvd.DistributedGradientTape(tape_1)
    
    grads = tape_1.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(
        zip(grads, discriminator.trainable_weights)
    )
    
    if first_batch:
        hvd.broadcast_variables(discriminator.variables, root_rank=0)
        hvd.broadcast_variables(d_optimizer.variables(), root_rank=0)
    
    # Update metrics
    d_loss_metric.update_state(d_loss)
    
    print("discriminator_2")
    
    # Train Discriminator with fake labels
    with tf.GradientTape() as tape_2:
        generated_images = generator(random_latent_vectors)
        fake_output = discriminator(generated_images)
        fake_targets = tf.zeros_like(fake_output)
        # Perform Label smoothing - assign a random integer in range [0.0, 0.3] for positive class
        fake_targets = tf.cast(fake_targets, dtype = 'float32') + 0.3*tf.random.uniform(tf.shape(fake_output), minval=0, maxval=1)
        d_loss = loss_fn(fake_targets, fake_output)
    
    # Horovod: add Horovod Distributed GradientTape.
    tape_2 = hvd.DistributedGradientTape(tape_2)    
    
    grads = tape_2.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(
        zip(grads, discriminator.trainable_weights)
    )
    
    if first_batch:
        hvd.broadcast_variables(discriminator.variables, root_rank=0)
        hvd.broadcast_variables(d_optimizer.variables(), root_rank=0)
        
    # Update metrics
    d_loss_metric.update_state(d_loss)
    
    print("generator")
    
    # Train the generator with real labels
    with tf.GradientTape() as tape_3:
        fake_output = discriminator(generator(random_latent_vectors))
        real_targets = tf.ones_like(real_output)
        g_loss = loss_fn(real_targets, fake_output)
        
    # Horovod: add Horovod Distributed GradientTape.
    tape_3 = hvd.DistributedGradientTape(tape_3)
    
    grads = tape_3.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    
    if first_batch:
        hvd.broadcast_variables(generator.variables, root_rank=0)
        hvd.broadcast_variables(g_optimizer.variables(), root_rank=0)

    # Update metrics
    d_loss_metric.update_state(d_loss)
    g_loss_metric.update_state(g_loss)
    return [d_loss_metric.result(), g_loss_metric.result()]

# Fit model
fit_start = time.perf_counter()

for epoch in range(epochs):
    
    # time epoch
    epoch_start = time.perf_counter()
    
    # Horovod: adjust number of steps based on number of GPUs.
    for batch, images in enumerate(dataset):
        loss_values = train_step(images, batch == 0)
    
        # if batch % 100 == 0 and hvd.local_rank() == 0:
        #     print('Step #%d\td_loss: %.6f\tg_loss: %.6f' % (batch, loss_values[0], loss_values[1]))
            
    epoch_end = time.perf_counter()
    epoch_time = epoch_end - epoch_start
    
    print('Epoch #%d\ttime taken: %.6f\td_loss: %.6f\tg_loss: %.6f' % (epoch, epoch_time, loss_values[0], loss_values[1]))
            
    random_latent_vectors = tf.random.normal(shape=(3, latent_dim))
    generated_images = generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(3):
        img = keras.preprocessing.image.array_to_img(generated_images[i])
        img.save("generated_img_%03d_%d.png" % (epoch, i))

fit_end = time.perf_counter();
fit_time = fit_end - fit_start

print('\n\tFit time: ', fit_time)