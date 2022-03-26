import random

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import os
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = 200
MODEL_PATH = './me.face.model.h5'


def load_dataset():
    label_name = {'me', 'other'}
    data = []
    label = []
    for index, name in enumerate(label_name):
        class_path = "./data/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = np.array(img.resize((200, 200)))
            data.append(img.reshape((3, 200, 200)))
            label.append(index)
    data = np.array(data)
    label = np.array(label)
    print(data.shape)
    print(label.shape)
    return data, label


def resize_image(image):
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image = image.resize((200, 200))
    image = np.array(image).reshape((3, 200, 200))
    return image


class Dataset:
    def __init__(self, path_name):
        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.path_name = path_name

        self.input_shape = None

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3):

        images, labels = load_dataset()

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        # if K.image_dim_ordering() == 'th':
        #     train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
        #     valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
        #     test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
        #     self.input_shape = (img_channels, img_rows, img_cols)
        # else:
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_rows, img_cols, img_channels)

        # 输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images  = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels  = test_labels


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2):
        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, padding='same',
                                     input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=20, nb_epoch=3, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)
        self.model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                           optimizer=sgd,
                           metrics=['accuracy'])

        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=True,             #是否使输入数据去中心化（均值为0），
                samplewise_center=False,             #是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  #是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  #是否将每个样本数据除以自身的标准差
                zca_whitening=False,                  #是否对输入数据施以ZCA白化
                rotation_range=20,                    #数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range= 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               #同上，只不过这里是垂直
                horizontal_flip = True,                 #是否进行随机水平翻转
                vertical_flip = False)                  #是否进行随机垂直翻转

            datagen.fit(dataset.train_images)

            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     steps_per_epoch = dataset.train_images.shape[0],
                                     epochs = nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def face_predict(self, image):
        #依然是根据后端系统确定维度顺序
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                             #尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   #与模型训练不同，这次只是针对1张图片进行预测
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        image = image.astype('float32')
        image /= 255

        result = self.model.predict_proba(image)
        print('result:', result)

        result = self.model.predict_classes(image)

        return result[0]


if __name__ == '__main__':
    dataset = Dataset('./data/')
    dataset.load()

    # 训练模型
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='./model/me.face.model.h5')

    model = Model()
    model.load_model(file_path='./model/me.face.model.h5')
    model.evaluate(dataset)
