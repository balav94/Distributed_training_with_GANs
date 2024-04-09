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

BATCH_SIZE = int(sys.argv[1])
# Set the number of epochs for trainining.
epochs = int(sys.argv[2])
IMG_SIZE = (64, 64)
IMG_SHAPE = (64, 64, 3)
    
file_name="../../celeba_gan/np_"+ str(1) +"_rank_"+ str(0) +".pickle"

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
    dataset[i] = np.array(data)/255.0
    
dataset = np.array(dataset)

dataset = dataset.reshape(dataset.shape[0], 64, 64, 3)

dataset = tf.data.Dataset.from_tensor_slices(dataset)

dataset = dataset.shuffle(10000).batch(BATCH_SIZE)

print('dataset :', dataset)

init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64*1, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64*2, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.LayerNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64*4, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.LayerNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(64*8, kernel_size=4, strides=2, padding="same", kernel_initializer=init),
        layers.LayerNormalization(),
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

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
g_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
d_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

d_loss_metric = keras.metrics.Mean(name="d_loss")
g_loss_metric = keras.metrics.Mean(name="g_loss")

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

def gradient_penalty(batch_size, real_images, fake_images):
    """ Calculates the gradient penalty.
    
    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = tf.cast(fake_images, dtype = 'float32') - tf.cast(real_images, dtype = 'float32')
    interpolated = tf.cast(real_images, dtype = 'float32') + alpha * diff
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = discriminator(interpolated, training=True)
    
    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step(real_images):
    if isinstance(real_images, tuple):
        real_images = real_images[0]
    
    # Get the batch size
    batch_size = tf.shape(real_images)[0]
    
    #The original paper recommends training
    # the discriminator for `x` more steps (typically 5) as compared to
    # one step of the generator. Here we will train it for 3 extra steps
    # as compared to 5 to reduce the training time.
    d_steps = 5
    gp_weight = 10
    
    # For each batch, we are going to perform the
    # following steps as laid out in the original paper:
    # 1. Train the generator and get the generator loss
    # 2. Train the discriminator and get the discriminator loss
    # 3. Calculate the gradient penalty
    # 4. Multiply this gradient penalty with a constant weight factor
    # 5. Add the gradient penalty to the discriminator loss
    # 6. Return the generator and discriminator losses as a loss dictionary
    
    # Train the discriminator first.
    for i in range(d_steps):
        # Get the latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, latent_dim)
        )
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = generator(random_latent_vectors, training=True)
            # Get the logits for the fake images
            fake_logits = discriminator(fake_images, training=True)
            # Get the logits for the real images
            real_logits = discriminator(real_images, training=True)
    
            # Calculate the discriminator loss using the fake and real image logits
            d_cost = discriminator_loss(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = gradient_penalty(batch_size, real_images, fake_images)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * gp_weight
    
        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        d_optimizer.apply_gradients(
            zip(d_gradient, discriminator.trainable_variables)
        )
        
        # Update metrics
        d_loss_metric.update_state(d_loss)
    
    # Train the generator
    # Get the latent vector
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    with tf.GradientTape() as tape:
        # Generate fake images using the generator
        generated_images = generator(random_latent_vectors, training=True)
        # Get the discriminator logits for fake images
        gen_img_logits = discriminator(generated_images, training=True)
        # Calculate the generator loss
        g_loss = generator_loss(gen_img_logits)
    
    # Get the gradients w.r.t the generator loss
    gen_gradient = tape.gradient(g_loss, generator.trainable_variables)
    # Update the weights of the generator using the generator optimizer
    g_optimizer.apply_gradients(
        zip(gen_gradient, generator.trainable_variables)
    )
    
    # Update metrics
    d_loss_metric.update_state(d_loss)
    g_loss_metric.update_state(g_loss)
    return [d_loss_metric.result(), g_loss_metric.result()]

fit_start = time.perf_counter()

for epoch in range(epochs):
    
    # time epoch
    epoch_start = time.perf_counter()
    
    # Horovod: adjust number of steps based on number of GPUs.
    for batch, images in enumerate(dataset):
        loss_values = train_step(images)
    
        if batch % 100 == 0:
            #and hvd.local_rank() == 0:
            print('Step #%d\td_loss: %.6f\tg_loss: %.6f' % (batch, loss_values[0], loss_values[1]))
            
    epoch_end = time.perf_counter()
    epoch_time = epoch_end - epoch_start
    
    print('Epoch #%d\ttime taken: %.6f\td_loss: %.6f\tg_loss: %.6f' % (epoch, epoch_time, loss_values[0], loss_values[1]))
            
    random_latent_vectors = tf.random.normal(shape=(3, latent_dim))
    generated_images = generator(random_latent_vectors)
    generated_images *= 255
    generated_images.numpy()
    for i in range(3):
        img = keras.preprocessing.image.array_to_img(generated_images[i])
        img.save("wgan_img_%03d_%d.png" % (epoch, i))


fit_end = time.perf_counter();
fit_time = fit_end - fit_start

print('\n\tFit time: ', fit_time)