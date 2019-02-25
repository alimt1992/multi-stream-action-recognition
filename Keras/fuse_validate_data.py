"""
Class for managing our data.
"""
import csv
import numpy as np
import os.path
import random
import threading
from keras.utils import to_categorical
import cv2

class DataSet():
    def __init__(self, class_limit=None, image_shape=(224, 224), original_image_shape=(341, 256), opt_flow_len=1, batch_size=16):
        """Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.class_limit = class_limit
        self.image_shape = image_shape
        self.original_image_shape = original_image_shape
       # self.n_snip = n_snip
        self.opt_flow_len = opt_flow_len
        self.batch_size = batch_size

        self.static_frame_path = os.path.join('/data/test')
        self.opt_flow_path = os.path.join('/data', 'opt_flow')
        self.pose_path = os.path.join('/data', 'pose')

        # Get the data.
        self.data_list = self.get_data_list()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()

        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        self.data_list = test
        
        # number of batches in 1 epoch
        self.n_batch = len(self.data_list) // self.batch_size

    @staticmethod
    def get_data_list():
        """Load our data list from file."""
        with open(os.path.join('/data', 'data_list.csv'), 'r') as fin:
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

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""

        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert label_hot.shape[0] == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data_list:
            if item[0] == 'test':
                test.append(item)
            else:
                train.append(item)
        return train, test

    def validation_generator(self):
        """Return a generator of optical frame stacks that we can use to test."""

        print("\nCreating validation generator with %d samples.\n" % len(self.data_list))

        idx = 0
        while 1:
            idx += 1
            idx = idx % self.n_batch
            print("\nGenerating batch number {0}/{1} ...".format(idx, self.n_batch))
            
            X_spatial_batch = []
            X_temporal_batch = []
            X_pose_batch = []
            y_batch = []

            # Get a list of batch-size samples.
            batch_list = self.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]

            for row in batch_list:
                # Get the stacked optical flows from disk.
                X_spatial, X_temporal, X_pose = self.get_static_frame_and_opt_flows_and_pose(row)
                
                # Get the corresponding labels
                y = self.get_class_one_hot(row[1])
                y = np.array(y)
                y = np.squeeze(y)

                X_spatial_batch.append(X_spatial)
                X_temporal_batch.append(X_temporal)
                X_pose_batch.append(X_pose)
                y_batch.append(y)

            X_batch = [np.array(X_spatial_batch), np.array(X_temporal_batch), np.array(X_pose_batch)]
            y_batch = np.array(y_batch)

            yield X_batch, y_batch

    def get_static_frame_and_opt_flows_and_pose(self, row):

        static_frame_dir = os.path.join(self.static_frame_path, row[1], row[2])
        opt_flow_dir = os.path.join(self.opt_flow_path, row[0], row[1], row[2])
        pose_dir = os.path.join(self.pose_path, row[0], row[1], row[2])


        # spatial parameters (crop at center for validation)
        left = int((self.original_image_shape[0] - self.image_shape[0]) * 0.5)
        top = int((self.original_image_shape[1] - self.image_shape[1]) * 0.5)
        right = left + self.image_shape[0]
        bottom = top + self.image_shape[1]


        # Get the static frame
        static_frame = cv2.imread(static_frame_dir + '.jpg')
        static_frame = static_frame / 255.0
        static_frame = cv2.resize(static_frame, self.image_shape)
        static_frame = np.array(static_frame)

        # Get the static frame
        opt_flow = cv2.imread(opt_flow_dir + '.png')
        opt_flow = opt_flow / 255.0
        opt_flow = cv2.resize(opt_flow, self.image_shape)
        opt_flow = np.array(opt_flow)

        # Get the static frame
        pose = cv2.imread(pose_dir + '.png')
        pose = pose / 255.0
        pose = cv2.resize(pose, self.image_shape)
        pose = np.array(pose)


        # Get the temporal frame
        #img = None # reset to be safe
        #img = cv2.imread(opt_flow_dir + '.png', 0)
        #img = np.array(img)
        #img = img - np.mean(img) # mean substraction
        #img = img[top: bottom, left: right]
        #img = img / 255.0 # normalize pixels
        #opt_flow = img
        #opt_flow = np.array(opt_flow)

        return static_frame, opt_flow, pose

