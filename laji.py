# -- coding:utf-8 --
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import glob, os, random,time
import numpy as np
from keras.models import load_model
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as n
from keras import backend as K
from PIL import Image
# K.tensorflow_backend._get_available_gpus()


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.backend import manual_variable_initialization
from keras.applications import VGG16

def processing_data(data_path, height, width, batch_size=4, validation_split=0.1):

    # :param data_path: 数据路径
    # :param height: 图片高度
    # :param width: 图片宽度
    # :param batch_size: batch: 每个批次的大小
    # :param validation_split: 测试集比例
    # :return: train_generator, validation_generator: 训练集、测试\验证集生成器
    

    # 训练集数据处理方式
    train_data = ImageDataGenerator(
            # 把[0,255]的像素值归一化成[0,1]的
            rescale=1. / 225,  
            # 下面的一系列操作都是因为我们目前的数据集不是很大，这些都是对图片进行一些骚操作，等于是加入一些噪声，比如旋转、切割等
            # 这样可以让我们的模型训练更加有效
            shear_range=0.1,  
            # ������ŵķ��ȣ���Ϊ�����������൱��[lower,upper] = [1 - zoom_range, 1+zoom_range]
            zoom_range=0.1,
            # ��������ͼƬ��ȵ�ĳ����������������ʱͼƬˮƽƫ�Ƶķ���?
            width_shift_range=0.1,
            # ��������ͼƬ�߶ȵ�ĳ����������������ʱͼƬ��ֱƫ�Ƶķ���
            height_shift_range=0.1,
            # ����ֵ���������ˮƽ���?
            horizontal_flip=True,
            # ����ֵ�����������ֱ���?
            vertical_flip=True,
            # �� 0 �� 1 ֮�両����������֤����ѵ�����ݵı���
            validation_split=validation_split  
    )

    # 测试集数据处理方式，这里就不进行骚操作了，只做归一化
    validation_data = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.1)

    train_generator = train_data.flow_from_directory(
            # 数据集路径
            data_path,
            # 图片宽和高
            target_size=(height, width),
            # batch大小，就是将完整的数据集分成一个个小小的batch扔进去训练
            batch_size=batch_size,
            # 标签类型，这里表示是多分类问题
            class_mode='categorical',
            # 读取数据后，选择是使用那一部分，这里选择的是训练集部分
            subset='training', 
            seed=0)
    validation_generator = validation_data.flow_from_directory(
            data_path,
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='categorical',

            # 这里选择验证集部分，实际上我们这里测试集和训练集已经合二为一了。。
            subset='validation',
            seed=0)

    return train_generator, validation_generator
# def processing_data(data_path):
#     """
#     ���ݴ���
#     :param data_path: ���ݼ�·��
#     :return: train, test:������ѵ�������ݡ����Լ�����
#     
#     # -------------------------- ʵ�����ݴ����ִ��� ----------------------------

#     # ------------------------------------------------------------------------
#     train_data, test_data = None, None
#     return train_data, test_data


def model(train_generator, test_generator, save_model_path='results/dnn.h5',
              log_dir="results/logs/"):
    # 初始化model Sequential的意思是初始化一个前馈神经网络结构，我们可以不断向其中添加神经网络的层（layer)
    model = Sequential()

    # 获取预训练模型 此处使用的是VGG16模型，这样不需要自行设计模型，就已经有了一个可靠的可以获取图片信息的卷积神经网络
    # input_shape由图片决定，由于我们的图片是384*512 RGB图片，所以输入维度为384*512*3
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(384, 512, 3))
    # 将VGG16模型设置为不进行训练，即始终使用网络的参数，后续为了提高效果可以将VGG16模型的参数也一并进行训练，但需注意对内存和显存的消耗会变大，如果不足可以考虑减小batch_size
    base_model.trainable = False

    # 将VGG16模型加入我们的模型
    model.add(base_model)

    # VGG16卷积神经网络会将输入的384*512*3数据 通过一系列操作（卷积、下采样、最大池化等，详情请百度卷积神经网络）
    # 将输入的数据转化成较小规模的数据，但依然是三维的，Flatten层的作用就是将三维的数据展开成1维的数据
    model.add(Flatten())

    # 添加一个全连接层用于训练，全连接层是神经网络最基本的结构，即形成多个线性的关系。activation是激活函数，通过激活函数可以将线性关系转化成可以有更强拟合
    # 能力的非线性函数
    model.add(Dense(256, activation='relu'))

    # dropout层的含义是随机在上一层得到的参数中舍弃一定比例的参数，以提高泛化能力
    model.add(Dropout(0.5))

    # 这也是一个全连接层，但激活函数不同，softmax函数专门用于多分类问题。此问题中将会输出6个值，表示该图片是这6个垃圾类别的概率
    model.add(Dense(6, activation='softmax'))

    # 输出model的结构信息
    model.summary()
              
    # adam = keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)

    # 编译模型，为模型设定训练时采用的优化函数、损失函数。这里优化函数是adam，损失函数是类别交叉熵。其实就是一种参数更新策略。
    # metrices表示在训练过程中需要输出的指标，帮助我们判断训练的效果，这里的accuracy表示类别判断的正确率，比如10个图片判断对了5个则
    # accuracy为0.5
    model.compile(
            # ���Ż���, ��Ҫ��Adam��sgd��rmsprop�ȷ�ʽ��
            optimizer='adam',
            # ��ʧ����,�������� categorical_crossentropy
            loss='categorical_crossentropy',
            # �ǳ�����ʧ����ֵ֮����ض�ָ��? ��������һ�㶼��׼ȷ��
            metrics=['accuracy'])

    # 这个本来是用来输出训练的图像的，后来没搞好，无视
    tensorboard = TensorBoard(log_dir)

    # 训练模型
    d = model.fit_generator(
            # 设置训练数据生成器，train_generator会不断按照一定的batch给模型提供训练数据（就是给几张图片拿给模型判断，根据判断结果优化参数）
            generator=train_generator,
            # epochs: 训练的周期数，即全部的数据会训练的轮数
            epochs=100,
            # 每个周期中的步数，就是批次数量（batch_num) = 训练集图片总数/每批次数量
            steps_per_epoch=2259 // 4,
            # 验证集，每个epoch结束后，把测试集拿来跑一下模型，根据指标判断一下这个epoch的训练效果，根据结果，判断是否要调整超参数（比如学习速率，dropout比例）
            validation_data=test_generator,
            # 验证集步数 测试集图片总数/每批次数量
            validation_steps=248 // 4,
            # 没啥用，无视
            callbacks=[tensorboard])
    # 模型训练结束后，保存模型
    model.save(save_model_path)

    # 用测试集来判断一下模型训练效果吧
    loss, accuracy = model.evaluate_generator(test_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    # -------------------------------------------------------------------------

    return model


def evaluate_mode(test_data, save_model_path):
    # 一个测试函数，可以用来读取已经训练出的模型，评估其效果
    # manual_variable_initialization(True)
    model = load_model(save_model_path)
    loss, accuracy = model.evaluate_generator(test_data)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
    # ---------------------------------------------------------------------------

from keras.models import load_model
from keras.preprocessing import image
import os


def predict(img, save_model_path):
    img = image.img_to_array(img)
    img = img / 255.
    img = img.reshape(1, 384, 512, 3)
    classes = ['cardboard','glass','metal','paper','plastic','trash']
    model = load_model(save_model_path)
    class_index = model.predict_classes(img)[0]
    return classes[class_index]


def main():

    # 数据集路径
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized"  # ���ݼ�·��
    # 结果储存路径，根据结果可以改一下
    save_model_path = './results/model.hdf5'  # ����ģ��·��������

    # 图片的高度和宽度
    height, width = 384, 512

    # 图片预处理，获取训练集和测试集的generator
    train_data, test_data = processing_data(data_path, height, width)

    # 建立模型并训练
    model(train_data, test_data, save_model_path)

    # 也可以选择加载我已经训练好的模型，记得将save_model_path的路径替换一下， 因为不需要训练了，可以把上面那一句注释掉

    # "results/model_15-0.94.hdf5"
    # 评估模型效果
    evaluate_mode(test_data, save_model_path)

    # 预测单个模型的类别
    img = Image.open('test.jpg', 'r')
    print(predict(img, save_model_path))


if __name__ == '__main__':
    main()

