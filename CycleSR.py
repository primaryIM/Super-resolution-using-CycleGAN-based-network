import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import sys
import time
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard

mode = 'inference'  # 'train' or 'inference'
main_dir = 'C:/'
beta_1 = 0.5
lr = 0.002
batch = 1
epochs = 2000


class LoadSave:

    @staticmethod
    def files_exist(load_model_dir_exist):
        sub_dirs = [s for s in os.listdir(load_model_dir_exist) if os.path.isdir(os.path.join(load_model_dir_exist, s))]
        sub_dirs.sort()
        if not sub_dirs:
            return []
        subdir = sub_dirs[-1]

        disc_a_path = os.path.join(load_model_dir_exist, subdir, 'model_disc_a')
        disc_b_path = os.path.join(load_model_dir_exist, subdir, 'model_disc_b')
        gen_a2b_path = os.path.join(load_model_dir_exist, subdir, 'model_gen_a2b')
        gen_b2a_path = os.path.join(load_model_dir_exist, subdir, 'model_gen_b2a')

        if os.path.isdir(disc_a_path) and os.path.isdir(disc_b_path) and \
                os.path.isdir(gen_a2b_path) and os.path.isdir(gen_b2a_path):
            return disc_a_path, disc_b_path, gen_a2b_path, gen_b2a_path, subdir
        return []

    @staticmethod
    def load_files(strategy, load_model_dir_files):
        files = LoadSave.files_exist(load_model_dir_files)
        if files:
            with strategy.scope():
                disc_a = tf.keras.models.load_model(files[0], compile=False)
                disc_b = tf.keras.models.load_model(files[1], compile=False)
                gen_a2b = tf.keras.models.load_model(files[2], compile=False)
                gen_b2a = tf.keras.models.load_model(files[3], compile=False)

                print('Models loaded from %s' % files[-1])
                return disc_a, disc_b, gen_a2b, gen_b2a
        else:
            return None

    @staticmethod
    def save_models(save_dir, disc_a, disc_b, gen_a2b, gen_b2a, epoch, gl_batch):
        subdir = 'E' + str(epoch) + 'B' + str(gl_batch)
        disc_a.save(os.path.join(save_dir, subdir, 'model_disc_a'), save_format='tf')
        disc_b.save(os.path.join(save_dir, subdir, 'model_disc_b'), save_format='tf')
        gen_a2b.save(os.path.join(save_dir, subdir, 'model_gen_a2b'), save_format='tf')
        gen_b2a.save(os.path.join(save_dir, subdir, 'model_gen_b2a'), save_format='tf')
        print('Models saved')
        return


class Residual(keras.Model):

    def __init__(self):
        super().__init__()
        self.convolution_1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.convolution_2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.scaling = layers.Lambda(lambda t: t * 0.1)

    def call(self, inputs, training=True):
        x = self.convolution_1(inputs)
        x = self.convolution_2(x)
        x = self.scaling(x)
        x = tf.math.add(x, inputs)
        return x


class Generator(keras.Model):

    def __init__(self):
        super().__init__()

        self.layer_1 = layers.Conv2D(filters=64, kernel_size=3, padding='same')

        self.res1 = Residual()
        self.res2 = Residual()
        self.res3 = Residual()
        self.res4 = Residual()
        self.res5 = Residual()
        self.res6 = Residual()
        self.res7 = Residual()
        self.res8 = Residual()

        self.layer_2 = layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.layer_3 = layers.Conv2D(filters=64 * (2 ** 2), kernel_size=3, padding='same')
        self.layer_4 = layers.Conv2D(filters=1, kernel_size=3, padding='same')

    def call(self, inputs, training=True):

        mean = tf.reduce_mean(inputs)
        unit = tf.math.subtract(inputs, mean)
        addendum = unit = self.layer_1(unit)

        unit = self.res1(unit)
        unit = self.res2(unit)
        unit = self.res3(unit)
        unit = self.res4(unit)
        unit = self.res5(unit)
        unit = self.res6(unit)
        unit = self.res7(unit)
        unit = self.res8(unit)

        unit = self.layer_2(unit)
        unit = tf.math.add(unit, addendum)
        unit = self.layer_3(unit)
        unit = tf.nn.depth_to_space(unit, 2)
        unit = self.layer_4(unit)
        unit = tf.math.add(unit, mean)
        return unit


class Generator2(keras.Model):

    def __init__(self):
        super().__init__()

        self.layer_1 = layers.Conv2D(filters=64, kernel_size=3, padding='same')

        self.res1 = Residual()
        self.res2 = Residual()
        self.res3 = Residual()
        self.res4 = Residual()
        self.res5 = Residual()
        self.res6 = Residual()
        self.res7 = Residual()
        self.res8 = Residual()

        self.layer_2 = layers.Conv2D(filters=64, kernel_size=3, padding='same')
        self.layer_3 = layers.Conv2D(filters=64 * (2 ** 2), strides=2, kernel_size=3, padding='same')
        self.layer_4 = layers.Conv2D(filters=1, kernel_size=3, padding='same')

    def call(self, inputs, training=True):
        mean = tf.reduce_mean(inputs)
        unit = tf.math.subtract(inputs, mean)
        addendum = unit = self.layer_1(unit)

        unit = self.res1(unit)
        unit = self.res2(unit)
        unit = self.res3(unit)
        unit = self.res4(unit)
        unit = self.res5(unit)
        unit = self.res6(unit)
        unit = self.res7(unit)
        unit = self.res8(unit)

        unit = self.layer_2(unit)
        unit = tf.math.add(unit, addendum)
        unit = self.layer_3(unit)
        unit = self.layer_4(unit)
        unit = tf.math.add(unit, mean)
        return unit


class Discriminator(keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.leaky = keras.layers.LeakyReLU(0.2)

        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.leaky(x)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.leaky(x)

        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = self.leaky(x)

        x = self.conv5(x)

        return x


class Image:

    @staticmethod
    def load_image(image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_bmp(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image * 2 - 1
        return image

    @staticmethod
    def generate_images(images, epoch, global_batch, main_dir_fr):
        plt.figure(figsize=(40, 20))

        for img_no, img in enumerate(images):
            a, b, a2b, b2a, ra, rb = img
            a = tf.reshape(a, [256, 256]).numpy()
            b = tf.reshape(b, [512, 512]).numpy()
            b2a = tf.reshape(b2a, [256, 256]).numpy()
            a2b = tf.reshape(a2b, [512, 512]).numpy()
            ra = tf.reshape(ra, [256, 256]).numpy()
            rb = tf.reshape(rb, [512, 512]).numpy()

            display_list = [a, b, a2b, b2a, ra, rb]
            title = ['A', 'B', 'A2B', 'B2A', 'RA', 'RB']

            for i in range(len(title)):
                plt.subplot(2, len(title), img_no * len(title) + i + 1)
                plt.title(title[i])

                plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
                plt.axis('off')

        pth = os.path.join(main_dir_fr, 'generated/E%d_B%d.png' % (epoch, global_batch))
        print(pth)
        plt.savefig(pth)
        plt.close()


class Pipeline:

    @staticmethod
    def create_pipeline(strategy, path_a, path_b, batch_size):

        ds_a = tf.data.Dataset.list_files(path_a + '/*.bmp')
        ds_b = tf.data.Dataset.list_files(path_b + '/*.bmp')

        ds_a = ds_a.map(lambda x: Image.load_image(x)).cache()
        ds_b = ds_b.map(lambda x: Image.load_image(x)).cache()

        len_a = len(os.listdir(path_a))
        len_b = len(os.listdir(path_b))
        if len_a > len_b:
            diff = len_b - len_b % batch_size
        else:
            diff = len_a - len_a % batch_size

        ds_a = ds_a.shuffle(buffer_size=len_a, reshuffle_each_iteration=True)
        ds_b = ds_b.shuffle(buffer_size=len_b, reshuffle_each_iteration=True)

        ds_a = ds_a.take(diff).batch(batch_size).prefetch(batch_size)
        ds_b = ds_b.take(diff).batch(batch_size).prefetch(batch_size)

        ds = tf.data.Dataset.zip((ds_a, ds_b))
        ds = strategy.experimental_distribute_dataset(ds)  # placing batch_size / 4 samples per GPU
        return ds

    @staticmethod
    def create_test_pipeline(path_a, path_b):
        ds_a = tf.data.Dataset.list_files(path_a + '/*.bmp', shuffle=False).take(2)
        ds_b = tf.data.Dataset.list_files(path_b + '/*.bmp', shuffle=False).take(2)

        ds_a = ds_a.map(lambda x: Image.load_image(x)).cache()
        ds_b = ds_b.map(lambda x: Image.load_image(x)).cache()

        ds_a = ds_a.batch(2).prefetch(2)
        ds_b = ds_b.batch(2).prefetch(2)

        ds = tf.data.Dataset.zip((ds_a, ds_b))
        return ds


class InferenceFramework:
    def __init__(self, main_dir_fr):

        self.inference_input = main_dir_fr + '/inference_input'
        self.inference_output = main_dir_fr + '/inference_output'
        self.load_models = main_dir_fr + '/model'
        self.strategy = tf.distribute.MirroredStrategy()

        models = LoadSave.load_files(self.strategy, self.load_models)
        if models:
            self.genA2B = models[2]
        else:
            print('No saved models found, terminating the execution')
            sys.exit()

    @tf.function
    def reconstruction(self, patches):

        _x = tf.zeros([1, 8704, 7680, 1], tf.float32)
        _y = tf.image.extract_patches(_x, [1, 512, 512, 1], [1, 512, 512, 1], [1, 1, 1, 1], 'SAME')
        grad = tf.gradients(_y, _x)[0]
        return tf.gradients(_y, _x, grad_ys=patches)[0] / grad

    def inference(self):

        ds = tf.data.Dataset.list_files(self.inference_input + '/*.bmp', shuffle=False)
        ds = ds.map(lambda x: Image.load_image(x)).cache()
        ds = ds.batch(1).prefetch(1)

        i = 0
        for batch_fr in ds:
            i += 1
            patch_size = [1, 256, 256, 1]
            batch_fr = tf.image.extract_patches(batch_fr, patch_size, patch_size, [1, 1, 1, 1], 'SAME')
            batch_fr = tf.reshape(batch_fr, [255, 256, 256, 1])
            main_output = tf.zeros([255, 512, 512, 1])
            for j in range(0, 255):
                slice1 = tf.slice(batch_fr, [j, 0, 0, 0], [1, 256, 256, 1])
                gen_a2b_output = self.genA2B(slice1, training=False)
                main_output = tf.tensor_scatter_nd_update(main_output, tf.constant([[j]]), gen_a2b_output)

            main_output = tf.cast(main_output, dtype=tf.float32)
            main_output = self.reconstruction(main_output)
            main_output = tf.reshape(main_output, [8704, 7680, 1])
            gen_a2b_output = tf.image.encode_png(tf.cast(main_output * 127 + 128, dtype=tf.uint8))
            pth = os.path.join(self.inference_output, 'output_%s.png' % i)
            tf.io.write_file(pth, gen_a2b_output)
        print('Output was successfully generated')


class TrainingFramework:
    def __init__(self, beta_1_fr, lr_fr, batch_fr, epochs_fr, main_dir_fr):

        self.main_dir = main_dir_fr
        self.trainA_path = main_dir_fr + '/train_LR'
        self.trainB_path = main_dir_fr + '/train_HR'
        self.testA_path = main_dir_fr + '/test_LR'
        self.testB_path = main_dir_fr + '/test_HR'
        self.models_dir = main_dir_fr + '/model'
        self.train_log_dir = main_dir_fr + '/logs'
        self.beta_1 = beta_1_fr
        self.learning_rate = lr_fr
        self.batch_size_per_tesla = batch_fr
        self.epochs = epochs_fr
        self.strategy = tf.distribute.MirroredStrategy()
        self.global_batch_size = batch_fr * self.strategy.num_replicas_in_sync
        self.global_batch_no = 0
        self.epoch = 0
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.train_loss_discriminatorA = tf.keras.metrics.Mean('train_discriminatorA_loss', dtype=tf.float32)
        self.train_loss_discriminatorB = tf.keras.metrics.Mean('train_discriminatorB_loss', dtype=tf.float32)
        self.train_loss_generatorA2B = tf.keras.metrics.Mean('train_generatorA2B_loss', dtype=tf.float32)
        self.train_loss_generatorB2A = tf.keras.metrics.Mean('train_generatorB2A_loss', dtype=tf.float32)
        self.train_loss_cycle = tf.keras.metrics.Mean('train_cycle_loss', dtype=tf.float32)
        self.train_loss_identityA2B = tf.keras.metrics.Mean('train_identity_loss_a2b', dtype=tf.float32)
        self.train_loss_identityB2A = tf.keras.metrics.Mean('train_identity_loss_b2a', dtype=tf.float32)
        self.train_loss_SSIMA2B = tf.keras.metrics.Mean('train_structural_loss_a2b', dtype=tf.float32)
        self.train_loss_SSIMB2A = tf.keras.metrics.Mean('train_SSIM_loss_B2A', dtype=tf.float32)

        models = LoadSave.load_files(self.strategy, self.models_dir)
        if models:
            self.discA = models[0]
            self.discB = models[1]
            self.genA2B = models[2]
            self.genB2A = models[3]
        else:
            self.discA = Discriminator()
            self.discB = Discriminator()
            self.genA2B = Generator()
            self.genB2A = Generator2()

        def lr_schedule():
            if self.epoch < 15:
                return self.learning_rate
            if self.epoch < 35:
                return 1e-1 * self.learning_rate
            if self.epoch < 50:
                return 1e-2 * self.learning_rate
            return 1e-3 * self.learning_rate

        self.discA_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1_fr)
        self.discB_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1_fr)
        self.genA2B_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1_fr)
        self.genB2A_opt = tf.keras.optimizers.Adam(lr_schedule, beta_1=beta_1_fr)

        self.ds = Pipeline.create_pipeline(self.strategy, self.trainA_path, self.trainB_path,
                                           batch_size=self.global_batch_size)
        self.ds_test = Pipeline.create_test_pipeline(self.testA_path, self.testB_path)

    @tf.function
    def train_step(self, dist_inputs):

        def discriminator_loss(disc_of_real_output, disc_of_gen_output, disc_lambda=1):
            real_loss = tf.reduce_sum(self.loss_obj(tf.ones_like(disc_of_real_output),
                                                    disc_of_real_output)) * (1. / self.global_batch_size)
            generated_loss = tf.reduce_sum(self.loss_obj(tf.zeros_like(disc_of_gen_output),
                                                         disc_of_gen_output)) * (1. / self.global_batch_size)
            total_disc_loss = (real_loss + generated_loss) / 2
            return total_disc_loss * disc_lambda

        def generator_loss(disc_of_gen_output, gen_lambda=0.001):
            gen_loss = tf.reduce_sum(self.loss_obj(tf.ones_like(disc_of_gen_output), disc_of_gen_output))
            return gen_loss * gen_lambda

        def cycle_consistency_loss(data_a, data_b, reconstructed_data_a, reconstructed_data_b, cyc_lambda=1):
            cyc_loss = tf.reduce_mean(tf.abs(data_a - reconstructed_data_a) + tf.abs(data_b - reconstructed_data_b))
            return cyc_loss * cyc_lambda

        def identity_loss(real_image, same_image, id_lambda=1):
            id_loss = tf.reduce_mean(tf.abs(real_image - same_image))
            return id_loss * id_lambda

        def structural_loss(real_image, generated_image, c1=0.02, c2=0.06, n=1, structural_lambda=1):
            ux = tf.math.reduce_mean(real_image)
            uy = tf.math.reduce_mean(generated_image)
            sigma_xy = tf.math.reduce_mean(tf.math.multiply(real_image, generated_image)) - ux * uy
            sigma_xx = tf.math.reduce_mean(tf.math.multiply(real_image, real_image)) - tf.math.reduce_mean(
                real_image) ** 2
            sigma_yy = tf.math.reduce_mean(tf.math.multiply(generated_image, generated_image)) - tf.math.reduce_mean(
                generated_image) ** 2

            st_loss = structural_lambda * (1 - ((ux * uy + c1) * (2 * sigma_xy + c2)) / (
                    (ux ** 2 + uy ** 2 + c1) * (sigma_xx ** 2 + sigma_yy ** 2 + c2))) / n

            return st_loss

        def upsample_loss(real_image, generated_image):
            generated_image = tf.image.resize(generated_image, [512,512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return tf.keras.losses.MSE(real_image, generated_image)

        def downsample_loss(real_image, generated_image):
            generated_image = tf.nn.avg_pool(generated_image, ksize = 3, strides = 2, padding = 'SAME')
            return tf.keras.losses.MSE(real_image, generated_image)

        def step_fn(inputs):
            input_a, input_b = inputs

            input_a = tf.image.random_crop(input_a, size=[1, 256, 256, 1])
            input_a_upsampled = tf.image.resize(input_a, [512, 512])

            input_b = tf.image.random_crop(input_b, size=[1, 512, 512, 1])
            input_b_downsampled = tf.image.resize(input_b, [256, 256])

            with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                    tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
                gen_a2b_output = self.genA2B(input_a, training=True)
                gen_b2a_output = self.genB2A(input_b, training=True)
                gen_b2a_output_upsampled = tf.image.resize(gen_b2a_output, [512, 512])

                same_a = self.genB2A(input_a_upsampled, training=True)
                same_a_upsampled = tf.image.resize(same_a, [512, 512])
                same_b = self.genA2B(input_b_downsampled, training=True)

                disc_a_real_output = self.discA(input_a_upsampled, training=True)
                disc_b_real_output = self.discB(input_b, training=True)

                disc_a_fake_output = self.discA(gen_b2a_output_upsampled, training=True)
                disc_b_fake_output = self.discB(gen_a2b_output, training=True)

                reconstructed_a = self.genB2A(gen_a2b_output, training=True)
                reconstructed_a_upsampled = tf.image.resize(reconstructed_a, [512, 512])
                reconstructed_b = self.genA2B(gen_b2a_output, training=True)

                disc_a_loss = discriminator_loss(disc_a_real_output, disc_a_fake_output)
                disc_b_loss = discriminator_loss(disc_b_real_output, disc_b_fake_output)

                cycle_loss = cycle_consistency_loss(input_a_upsampled, input_b,
                                                    reconstructed_a_upsampled, reconstructed_b)

                generator_loss_a2b = generator_loss(disc_b_fake_output)
                identity_loss_a2b = identity_loss(input_b, same_b)
                structural_loss_a2b = structural_loss(input_a_upsampled, gen_a2b_output, structural_lambda=2)
                downsample_loss_a2b = downsample_loss(input_a, gen_a2b_output)
                gen_a2b_loss = (generator_loss_a2b + cycle_loss + identity_loss_a2b +
                                downsample_loss_a2b + structural_loss_a2b) * (1.0 / self.global_batch_size)

                generator_loss_b2a = generator_loss(disc_a_fake_output)
                identity_loss_b2a = identity_loss(input_a_upsampled, same_a_upsampled)
                structural_loss_b2a = structural_loss(input_b, gen_b2a_output_upsampled)
                upsample_loss_b2a = upsample_loss(input_b, gen_b2a_output)
                gen_b2a_loss = (generator_loss_b2a + cycle_loss + identity_loss_b2a +
                                upsample_loss_b2a + structural_loss_b2a) * (1.0 / self.global_batch_size)

            gen_a2b_gradients = genA2B_tape.gradient(gen_a2b_loss, self.genA2B.trainable_variables)
            gen_b2a_gradients = genB2A_tape.gradient(gen_b2a_loss, self.genB2A.trainable_variables)

            disc_a_gradients = discA_tape.gradient(disc_a_loss, self.discA.trainable_variables)
            disc_b_gradients = discB_tape.gradient(disc_b_loss, self.discB.trainable_variables)

            self.genA2B_opt.apply_gradients(zip(gen_a2b_gradients, self.genA2B.trainable_variables))
            self.genB2A_opt.apply_gradients(zip(gen_b2a_gradients, self.genB2A.trainable_variables))

            self.discA_opt.apply_gradients(zip(disc_a_gradients, self.discA.trainable_variables))
            self.discB_opt.apply_gradients(zip(disc_b_gradients, self.discB.trainable_variables))

            self.train_loss_discriminatorA(disc_a_loss)
            self.train_loss_discriminatorB(disc_b_loss)
            self.train_loss_generatorA2B(generator_loss_a2b)
            self.train_loss_generatorB2A(generator_loss_b2a)
            self.train_loss_cycle(cycle_loss)
            self.train_loss_identityA2B(identity_loss_a2b)
            self.train_loss_identityB2A(identity_loss_b2a)
            self.train_loss_SSIMA2B(structural_loss_a2b)
            self.train_loss_SSIMB2A(structural_loss_b2a)

            return

        self.strategy.run(step_fn, args=(dist_inputs,))

        return

    def inference(self):
        for batch_in in self.ds_test:
            input_a, input_b = batch_in

            gen_a2b_output = self.genA2B(input_a, training=False)
            gen_b2a_output = self.genB2A(input_b, training=False)

            reconstructed_a = self.genB2A(gen_a2b_output, training=False)
            reconstructed_b = self.genA2B(gen_b2a_output, training=False)

            images = tf.data.Dataset.from_tensor_slices((input_a, input_b, gen_a2b_output,
                                                         gen_b2a_output, reconstructed_a, reconstructed_b))

            Image.generate_images(images, self.epoch, self.global_batch_no, self.main_dir)

            break

    def train(self):
        print('Start training on %d device(s)' % self.strategy.num_replicas_in_sync)

        training_start = time.time()
        for self.epoch in range(self.epochs):
            epoch_start = time.time()
            self.global_batch_no = 0
            for distributed_batch in self.ds:
                self.global_batch_no += 1
                batch_start = time.time()
                with self.strategy.scope():
                    self.train_step(distributed_batch)
                print('Time taken for epoch {} and global batch {} is {} sec'.format(self.epoch + 1,
                                                                                     self.global_batch_no,
                                                                                     time.time() - batch_start))
                print('Image throughput: %.3f images/s' % (self.global_batch_size / (time.time() - batch_start)))

                if self.global_batch_no % 100 == 0:
                    self.inference()
            if self.epoch % 10 == 0 and self.epoch != 0:
                LoadSave.save_models(self.models_dir, self.discA, self.discB,
                                     self.genA2B, self.genB2A, self.epoch, self.global_batch_no)

            with self.train_summary_writer.as_default():
                tf.summary.scalar('Discriminator A loss', self.train_loss_discriminatorA.result(), step=self.epoch)
                tf.summary.scalar('Discriminator B loss', self.train_loss_discriminatorB.result(), step=self.epoch)
                tf.summary.scalar('Generator A2B loss', self.train_loss_generatorA2B.result(), step=self.epoch)
                tf.summary.scalar('Generator B2A loss', self.train_loss_generatorB2A.result(), step=self.epoch)
                tf.summary.scalar('Cycle consistency loss', self.train_loss_cycle.result(), step=self.epoch)
                tf.summary.scalar('Identity loss A2B', self.train_loss_identityA2B.result(), step=self.epoch)
                tf.summary.scalar('Identity loss B2A', self.train_loss_identityB2A.result(), step=self.epoch)
                tf.summary.scalar('Structural loss A2B', self.train_loss_SSIMA2B.result(), step=self.epoch)
                tf.summary.scalar('Structural loss B2A', self.train_loss_SSIMB2A.result(), step=self.epoch)

            self.train_loss_discriminatorA.reset_states()
            self.train_loss_discriminatorB.reset_states()
            self.train_loss_generatorA2B.reset_states()
            self.train_loss_generatorB2A.reset_states()
            self.train_loss_cycle.reset_states()
            self.train_loss_identityA2B.reset_states()
            self.train_loss_identityB2A.reset_states()
            self.train_loss_SSIMA2B.reset_states()
            self.train_loss_SSIMB2A.reset_states()

            print('Time taken for whole epoch {} is {} sec\n'.format(self.epoch + 1, time.time() - epoch_start))

        print('Time taken for whole training is {} sec\n'.format(time.time() - training_start))


if mode == 'train':
    fw = TrainingFramework(beta_1, lr, batch, epochs, main_dir)
    fw.train()
elif mode == 'inference':
    fw = InferenceFramework(main_dir)
    fw.inference()
else:
    print('Please change "mode" variable to available one')
