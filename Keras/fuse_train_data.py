"""
Class for managing our data.
"""
import csv
import numpy as np
import os.path
import random
import threading
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import cv2

class DataSet():
    def __init__(self, class_limit=None, image_shape=(224, 224), original_image_shape=(341, 256), batch_size=16):
        """Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.class_limit = class_limit
        self.image_shape = image_shape
        self.original_image_shape = original_image_shape
        self.batch_size = batch_size

        self.static_frame_path = os.path.join('./data','test')
        self.opt_flow_path = os.path.join('./data', 'opt_flow')
        self.pose_path = os.path.join('./data', 'pose')

        # Get the data.
        self.data_list = self.get_data_list()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()


        # number of batches in 1 epoch
        self.n_batch_train = len(train) // self.batch_size
        self.n_batch_test = len(test) // self.batch_size

    @staticmethod
    def get_data_list():
        """Load our data list from file."""
        with open(os.path.join('./data', 'data_list.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data_list = list(reader)

        return data_list

    def clean_data_list(self):
        data_list_clean = []
        for item in self.data_list:
            if item[1] in self.classes:
                data_list_clean.append(item)

        return data_list_clean

    def get_classes(self):
        """Extract the classes from our data, '\n'. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data_list:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes




    def format_gen_outputs(gen1,gen2,gen3):
        x1 = gen1[0]
        x2 = gen2[0]
        x3 = gen3[0]
        y1 = gen1[1]
        return [x1, x2, x3], y1


    def get_generators(self):

        train_datagen = ImageDataGenerator(
                preprocessing_function=self.normalize,
                rotation_range=30,
                rescale=1./255,
                shear_range=0.2,
                horizontal_flip=True
                # width_shift_range=0.2,
                # height_shift_range=0.2)
        )

        valid_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=self.normalize)


        valid_dir1=os.path.join('./data', 'valid')
        valid_dir2=os.path.join('./data/opt_flow', 'valid')
        valid_dir3=os.path.join('./data/pose', 'valid')

        train_dir1=os.path.join('./data', 'train')
        train_dir2=os.path.join('./data/opt_flow', 'train')
        train_dir3=os.path.join('./data/pose', 'train')

        seed = random.randint(1,1001)
        valid_genX1  = valid_datagen.flow_from_directory(valid_dir1,
                                               target_size=self.image_shape,
                                               batch_size=self.batch_size,
                                               #classes=data.classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                                seed=seed)

        valid_genX2  = valid_datagen.flow_from_directory(valid_dir2,
                                               target_size=self.image_shape,
                                               batch_size=self.batch_size,
                                               #classes=data.classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                                seed=seed)

        valid_genX3  = valid_datagen.flow_from_directory(valid_dir3,
                                               target_size=self.image_shape,
                                               batch_size=self.batch_size,
                                               #classes=data.classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                                seed=seed)

        validation_generator = map(self.format_gen_outputs, valid_genX1, valid_genX2, valid_genX3)

        seed = random.randint(1,1001)

        train_genX1  = train_datagen.flow_from_directory(train_dir1,
                                               target_size=self.image_shape,
                                               batch_size=self.batch_size,
                                               #classes=data.classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                                seed=seed)

        train_genX2  = train_datagen.flow_from_directory(train_dir2,
                                               target_size=self.image_shape,
                                               batch_size=self.batch_size,
                                               #classes=data.classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                                seed=seed)

        train_genX3  = train_datagen.flow_from_directory(train_dir3,
                                               target_size=self.image_shape,
                                               batch_size=self.batch_size,
                                               #classes=data.classes,
                                               class_mode='categorical',
                                               shuffle=True,
                                                seed=seed)

        train_generator = map(self.format_gen_outputs, train_genX1, train_genX2, train_genX3)


        return train_generator, validation_generator

    def normalize(self, x):

        x[1] = (x[1]-0.458)/0.229
        x[2] = (x[2]-0.456)/0.224
        x[3] = (x[3]-0.406)/0.225

        return x