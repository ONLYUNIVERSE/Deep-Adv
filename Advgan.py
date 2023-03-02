!pip install -q -U pip==20.2
!pip install tensorflow-addons
!pip install git+https://github.com/google-research/tensorflow_constrained_optimization
!pip install -q tensorflow-datasets tensorflow
!pip install fairness-indicators \
  "absl-py==0.12.0" \
  "apache-beam<3,>=2.36" \
  "avro-python3==1.9.1" \
  "pyzmq==17.0.0"
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/MachineLearning/celebaimages")
import os
import sys
import cv2
import glob
import shutil
import urllib
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from PIL import Image
from tensorflow import keras
from imageio import imread, imsave
from tensorflow.keras import layers
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
%matplotlib inline
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

PATH = os.path.join(os.path.dirname('./7'), './7')

image_files = os.listdir(PATH)
image_files.sort(key = lambda x:int (x[:-4]))

print("over")

def load_image(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize_with_pad(image, 224, 224, 'nearest', antialias=True)
  image = tf.cast(image,dtype=tf.float32)
  image = image / 255
  image = tf.convert_to_tensor(image)
  return image
train_data = []
for x in image_files:
  print(x)
  #train_data = tf.data.Dataset.list_files('./1/'+x)
  #train_data = train_data.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_data.append(load_image('./7/'+x))
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
from tensorflow.python.client import device_lib

# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
# 打印
#     print(local_device_protos)

# 只打印GPU设备
[print(x) for x in local_device_protos if x.device_type == 'GPU']
train_data = tf.stack(train_data, axis=0)
for i in range(10):
    plt.subplot(1, 10, i + 1)

    pixels = np.array(train_data[i, :, :])
    plt.imshow(pixels)
    plt.axis('off')
plt.show()
f = open("./list_attr_celeba.txt")
train_label = []
x1 = 0
x2 = 1
test_label = []
line = f.readline()
line = f.readline()
line = f.readline()
#while line:
for x in range(4500):
    array = line.split()
    if (array[21] == "-1"):
      train_label.append(x1)
    else:
      train_label.append(x2)
    #if x > 19999:
    #  if (array[21] == "-1"):
    #      test_label.append(x1)
    #  else:
    #      test_label.append(x2)
    line = f.readline()
f.close()
train_label=tf.reshape(train_label,[4500])
#test_label=tf.reshape(test_label,[10000])
print(train_label)
#print(test_label)
#train_image = train_data / 255
train_image = np.expand_dims(train_data, -1)
#train_image = tf.cast(train_image,tf.int8)
train_label = tf.cast(train_label,tf.int8)
#test_image = test_data / 255
#test_image = np.expand_dims(test_data, -1)
train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
#test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label))
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 10
train_image_count = 4500#train_data.shape[0]

print(train_image_count,train_label.shape)
#test_image_count = test_image.shape[0]
train_dataset = train_dataset.shuffle(train_image_count).batch(BATCH_SIZE)
#test_dataset = test_dataset.shuffle(test_image_count).batch(BATCH_SIZE)
dloss = 0
gloss = 0


def generator_model():
    seed = layers.Input(shape=((224, 224, 3)))
    label = layers.Input(shape=(()))
    print(seed.shape)
    x = layers.Conv2D(8, (4, 4), strides=(2, 2), padding='same', use_bias=True)(seed)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=True)(x)
    x = layers.Activation('tanh')(x)

    x = x + seed

    model = tf.keras.Model(inputs=[seed, label], outputs=x)

    return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    name_base = str(stage) + block + '_identity_block_'

    x = layers.Conv2D(filters1, (1, 1), name=name_base + 'conv1')(input_tensor)
    x = layers.BatchNormalization(name=name_base + 'bn1')(x)
    x = layers.Activation('relu', name=name_base + 'relu1')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=name_base + 'conv2')(x)
    x = layers.BatchNormalization(name=name_base + 'bn2')(x)
    x = layers.Activation('relu', name=name_base + 'relu2')(x)

    x = layers.Conv2D(filters3, (1, 1), name=name_base + 'conv3')(x)
    x = layers.BatchNormalization(name=name_base + 'bn3')(x)

    x = layers.add([x, input_tensor], name=name_base + 'add')
    x = layers.Activation('relu', name=name_base + 'relu4')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    res_name_base = str(stage) + block + '_conv_block_res_'
    name_base = str(stage) + block + '_conv_block_'

    x = layers.Conv2D(filters1, (1, 1), strides=strides, name=name_base + 'conv1')(input_tensor)
    x = layers.BatchNormalization(name=name_base + 'bn1')(x)
    x = layers.Activation('relu', name=name_base + 'relu1')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=name_base + 'conv2')(x)
    x = layers.BatchNormalization(name=name_base + 'bn2')(x)
    x = layers.Activation('relu', name=name_base + 'relu2')(x)

    x = layers.Conv2D(filters3, (1, 1), name=name_base + 'conv3')(x)
    x = layers.BatchNormalization(name=name_base + 'bn3')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, name=res_name_base + 'conv')(input_tensor)
    shortcut = layers.BatchNormalization(name=res_name_base + 'bn')(shortcut)

    x = layers.add([x, shortcut], name=name_base + 'add')
    x = layers.Activation('relu', name=name_base + 'relu4')(x)
    return x


def discriminator_model(input_shape=[224, 224, 3], classes=2):
    img_input = layers.Input(shape=[224, 224, 3])
    print(img_input.shape)
    x = layers.ZeroPadding2D((3, 3))(img_input)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    x = layers.Flatten()(x)

    x1 = layers.Dense(1)(x)

    x = layers.Dense(2, activation='softmax', name='fc1000')(x)  # 分类输出
    model = tf.keras.Model(inputs=img_input, outputs=[x])
    # model = Model(img_input, x, name='resnet50')

    # 加载预训练模型
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


def f_model(input_shape=[224, 224, 3], classes=2):
    img_input = layers.Input(shape=[224, 224, 3])
    print(img_input.shape)
    x = layers.ZeroPadding2D((3, 3))(img_input)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    x = layers.Flatten()(x)

    x1 = layers.Dense(1)(x)

    x = layers.Dense(2, activation='softmax', name='fc1000')(x)  # 分类输出
    model = tf.keras.Model(inputs=img_input, outputs=[x])
    # model = Model(img_input, x, name='resnet50')

    # 加载预训练模型
    # model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model
generator = generator_model()
discriminator = discriminator_model()
functionor = f_model()
discriminator.load_weights('./resnet50checkpoint')
#functionor.load_weights('./resnet50checkpoint')
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # 真假损失
category_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

def discriminator_loss(real_cat_out,fake_cat_out, label): # 接收真图 和 真实图片的分类  假图 加label
    real_loss = category_cross_entropy(tf.ones_like(label),real_cat_out)
    fake_loss = category_cross_entropy(tf.zeros_like(label),real_cat_out)
    return fake_loss+real_loss
def generator_loss(real_cat_out,fake_cat_out, label):
    real_loss = category_cross_entropy(tf.ones_like(label),real_cat_out)
    cat_loss = category_cross_entropy(tf.zeros_like(label), fake_cat_out)
    return cat_loss+real_loss
generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
functionor_optimizer = tf.keras.optimizers.Adam(1e-5)
loss_history = []
dloss_history = []


def train_step(images, labels, gloss, dloss):
    # print(labels.shape)
    batchsize = labels.shape[0]
    # noise = tf.random.normal([batchsize, noise_dim])
    # test_noise_input = tf.squeeze(images, axis=-1)
    # for i in range(10):
    #  plt.subplot(1, 10, i+1)
    #
    #  pixels = np.array(test_noise_input[i, :, :])
    #
    #  plt.imshow(pixels)
    #  plt.axis('off')
    # disi = labels.numpy()
    # print("pre=",disi[0],disi[1],disi[2],disi[3],disi[4],disi[5],disi[6],disi[7],disi[8],disi[9])
    # plt.show()

    with tf.GradientTape() as gen_tape, tf.GradientTape() as func_tape:
        generated_images = generator((images, labels), training=True)

        real_out = functionor(images, training=True)  # 真图真假结果
        fake_out = functionor(generated_images, training=True)  # 假图真假结果
        fake_cat_out = discriminator(generated_images, training=True)  # 假图分类结果

        generated_imagess = tf.cast(generated_images, dtype=tf.float32)
        imagess = tf.cast(images, dtype=tf.float32)
        imagess = tf.squeeze(imagess, axis=-1)
        dif_loss = tf.reduce_mean(tf.square(imagess - generated_imagess))
        gen_loss1 = generator_loss(fake_out, fake_cat_out, labels)
        gen_loss = gen_loss1 + 10 * dif_loss
        # gen_loss = gen_loss1
        disc_loss = discriminator_loss(real_out, fake_out, labels)
    gloss = gen_loss1
    dloss = disc_loss
    # gloss = 0
    # dloss = disc_loss
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_functionor = func_tape.gradient(disc_loss, functionor.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    functionor_optimizer.apply_gradients(zip(gradients_of_functionor, functionor.trainable_variables))
    loss_history.append(gen_loss.numpy().mean())
    dloss_history.append(disc_loss.numpy().mean())
    return dloss, gloss


from matplotlib import test


def generate_and_save_images(model, test_noise_input, test_cat_input, epoch):
    print('Epoch:', epoch + 1)

    predictions = model((test_noise_input, test_cat_input), training=False)

    test_noise_input = tf.squeeze(test_noise_input, axis=-1)
    for i in range(10):
        # plt.figure(1)
        # plt.imshow(np.array(test_noise_input[i,:,:]))
        # plt.axis('off')
        # plt.show()
        plt.subplot(1, 10, i + 1)
        pixels = np.array(test_noise_input[i, :, :])
        plt.imshow(pixels)
        plt.axis('off')
    plt.show()

    generated_imagess = tf.cast(predictions, dtype=tf.float32)
    imagess = tf.cast(test_noise_input, dtype=tf.float32)
    mask = imagess - generated_imagess
    mask = tf.squeeze(mask)

    for i in range(10):
        # plt.figure(1)
        # plt.imshow(np.array(mask[i,:,:]))
        # plt.axis('off')
        # plt.show()
        plt.subplot(1, 10, i + 1)
        pixels = np.array(mask[i, :, :])
        plt.imshow(pixels)
        plt.axis('off')
    plt.show()

    predictions = tf.squeeze(predictions)

    psnr1 = tf.image.psnr(predictions, test_noise_input, max_val=1.0)
    print(psnr1)

    fig = plt.figure(figsize=(10, 1))
    disc = discriminator(predictions, training=False)
    # disc = discriminator(test_noise_input,training=False)
    dis = tf.nn.softmax(disc)
    dis = tf.argmax(dis, axis=1)
    dis = dis.numpy()
    print("pre=", dis[0], dis[1], dis[2], dis[3], dis[4], dis[5], dis[6], dis[7], dis[8], dis[9])
    disi = test_cat_input.numpy()
    print("real=", disi[0], disi[1], disi[2], disi[3], disi[4], disi[5], disi[6], disi[7], disi[8], disi[9])
    # test_noise_input = tf.squeeze(test_noise_input, axis=-1)
    print("gen")
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow((predictions[i, :, :] + 1) / 2, cmap='gray')
        # plt.imshow((test_noise_input[i, :, :].astype('uint8')))
        # pixels = np.array(test_noise_input[i, :, :])
        # pixels = np.array(predictions[i, :, :])
        # plt.imshow(pixels.astype('uint8'))
        # plt.imshow(pixels)
        plt.axis('off')
    # plt.figure(1)
    # plt.imshow(np.array(predictions[i,:,:]))
    # plt.axis('off')
    # plt.show()
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        # print(epoch)
        for image_batch, label_batch in dataset:
            ddloss, ggloss = train_step(image_batch, label_batch, dloss, gloss)
        if epoch % 10 == 0:
            # print(image_batch.shape)
            generate_and_save_images(generator, image_batch, label_batch, epoch)

            # discriminator.save_weights('./testcheckpoint')
            # generator.save_weights('./gtestcheckpoint')
            plt.plot(loss_history)
            plt.xlabel('Batch #')
            plt.ylabel('Loss [entropy]')
            plt.show()
            plt.plot(dloss_history)
            plt.xlabel('Batch #')
            plt.ylabel('dLoss [entropy]')
            plt.show()
        # gen_loss = 300*ggloss + 100*ddloss
        # gen_loss = ggloss+10*ddloss
        print("gloss=", ggloss, "dloss=", ddloss)
        # loss_history.append(gen_loss.numpy().mean())

    generate_and_save_images(generator,
                             image_batch,
                             label_batch,
                             epoch)

EPOCHS = 50
train(train_dataset, EPOCHS)
functionor.save_weights('./advganfuncheckpoint')
generator.save_weights('./advgangencheckpoint')


def generate_and_save_image(model, test_noise_input, test_cat_input, epoch):
    print('Epoch:', epoch + 1)

    predictions = model((test_noise_input, test_cat_input), training=False)

    test_noise_input = tf.squeeze(test_noise_input, axis=-1)
    for i in range(10):
        plt.figure(1)
        plt.imshow(np.array(test_noise_input[i, :, :]))
        plt.axis('off')
        plt.show()
        # plt.subplot(1, 10, i+1)
        # pixels = np.array(test_noise_input[i, :, :])
        # plt.imshow(pixels)
        # plt.axis('off')
        # plt.show()

    generated_imagess = tf.cast(predictions, dtype=tf.float32)
    imagess = tf.cast(test_noise_input, dtype=tf.float32)
    mask = imagess - generated_imagess
    mask = tf.squeeze(mask)

    for i in range(10):
        plt.figure(1)
        plt.imshow(np.array(mask[i, :, :]))
        plt.axis('off')
        plt.show()
        # plt.subplot(1, 10, i+1)
        # pixels = np.array(mask[i, :, :])
        # plt.imshow(pixels)
        # plt.axis('off')
    # plt.show()

    predictions = tf.squeeze(predictions)

    psnr1 = tf.image.psnr(predictions, test_noise_input, max_val=1.0)
    print(psnr1)

    fig = plt.figure(figsize=(10, 1))
    disc = discriminator(predictions, training=False)
    # disc = discriminator(test_noise_input,training=False)
    dis = tf.nn.softmax(disc)
    dis = tf.argmax(dis, axis=1)
    dis = dis.numpy()
    print("pre=", dis[0], dis[1], dis[2], dis[3], dis[4], dis[5], dis[6], dis[7], dis[8], dis[9])
    disi = test_cat_input.numpy()
    print("real=", disi[0], disi[1], disi[2], disi[3], disi[4], disi[5], disi[6], disi[7], disi[8], disi[9])
    # test_noise_input = tf.squeeze(test_noise_input, axis=-1)
    print("gen")
    for i in range(10):
        # plt.subplot(1, 10, i+1)
        # plt.imshow((predictions[i, :, :] + 1)/2, cmap='gray')
        # plt.imshow((test_noise_input[i, :, :].astype('uint8')))
        # pixels = np.array(test_noise_input[i, :, :])
        # pixels = np.array(predictions[i, :, :])
        # plt.imshow(pixels.astype('uint8'))
        # plt.imshow(pixels)
        # plt.axis('off')
        plt.figure(1)
        plt.imshow(np.array(predictions[i, :, :]))
        plt.axis('off')
        plt.show()
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def test(dataset, epochs):
    time = 0
    total_correct = 0
    total_number = 0
    correctrate = tf.zeros([2, 2], tf.int32)  #
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            predictions = generator((image_batch, label_batch), training=False)
            predictions = tf.squeeze(predictions)

            fig = plt.figure(figsize=(10, 1))
            logits = discriminator(predictions, training=False)

            generate_and_save_image(generator, image_batch, label_batch, epoch)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)

            time += 1
            for k in range(10):
                if time % 40 == 0:
                    break

                y1 = tf.gather_nd(label_batch, [k]).numpy()
                p1 = tf.gather_nd(pred, [k]).numpy()
                # print(times1,'y1=',y1,p1,'k=',k)
                # val1=gather_nd(correctrate,[y1,p1])
                # val=val1.numpy()
                val = 1
                # val=1
                index1 = [(y1, p1)]
                update1 = [val]
                shape = tf.constant([2, 2])
                # print(index1,update1)
                correctrate2 = tf.scatter_nd(index1, update1, shape)
                # ttet=correctrate2.numpy()
                # print(ttet)
                correctrate = correctrate2 + correctrate
            label_batch1 = tf.cast(label_batch, dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, label_batch1), dtype=tf.int32))
            total_correct += int(correct)
            total_number += image_batch.shape[0]

            acc = total_correct / total_number
            testrate = correctrate.numpy()
            print(testrate)
            print("acc=", acc, "total=", total_number)
test(train_dataset,1)