import numpy as np
#from keras_resnet101 import resnet101_model as Resnet101
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, load_model, Model
from keras.layers import Input, average, concatenate, GlobalAveragePooling2D
from keras.layers import TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

class ResearchModels():
    def __init__(self, nb_classes, image_shape = (224, 224), saved_model=None, saved_temporal_weights=None, saved_spatial_weights=None, saved_pose_weights=None):

        """
        `nb_classes` = the number of classes to predict
        `opt_flow_len` = the length of optical flow frames
        `image_shape` = shape of image frame
        `saved_model` = the path to a saved Keras model to load
        """
        self.nb_classes = nb_classes
        self.load_model = load_model
        self.saved_model = saved_model
        self.saved_pose_weights = saved_pose_weights
        self.saved_temporal_weights = saved_temporal_weights
        self.saved_spatial_weights = saved_spatial_weights

        self.input_shape_spatial = (image_shape[0], image_shape[1], 3)
        self.input_shape_temporal = (image_shape[0], image_shape[1], 3)
        self.input_shape_pose = (image_shape[0], image_shape[1], 3)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Load model
        # If saved fuse model exists, directly load
        if self.saved_model is not None: 
            print("\nLoading model %s" % self.saved_model)
            self.model = self.three_stream_fuse()
            self.model.load_weights(self.saved_model)
        # Otherwise build the model and load weights for both streams
        else: 
            #print("\nLoading the two-stream model...")
            #self.model = self.two_stream_fuse()

            print("\nLoading the three-stream model...")
            self.model = self.three_stream_fuse()

        optimizer = Adam( lr=0.0002)
#        optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    # Two-stream fused model
    def two_stream_fuse(self):
        # spatial stream (frozen)
        cnn_spatial = self.cnn_spatial()

        # temporal stream (frozen)
        cnn_temporal = self.cnn_temporal()


        # fused by taking average
        outputs = average([cnn_spatial.output, cnn_temporal.output])

        model = Model([cnn_spatial.input, cnn_temporal.input], outputs)

        return model

    # Three-stream fused model
    def three_stream_fuse(self):
        # spatial stream (frozen)
        cnn_spatial = self.cnn_spatials()

        # temporal stream (frozen)
        cnn_temporal = self.cnn_temporal()

        # temporal stream (frozen)
        cnn_pose = self.cnn_pose()

        # fused by taking average
        outputs = average([cnn_spatial.output, cnn_temporal.output, cnn_pose.output])

        model = Model([cnn_spatial.input, cnn_temporal.input, cnn_pose.input], outputs)

        return model



    # CNN model for the spatial stream
    def cnn_spatial(self):
        #model
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #base_model = VGG16(weights=weights, include_top=False, input_shape=self.input_shape)
    
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation='softmax')(x)
    
        model = Model(inputs=base_model.input, outputs=predictions)

        # spatial stream (frozen)
        if self.saved_spatial_weights is None:
            print("[ERROR] No saved_spatial_weights weights file!")
        else:
            model.load_weights(self.saved_spatial_weights)
        for layer in model.layers:
            layer.trainable = False

        return model

    # CNN model for the temporal stream
    def cnn_temporal(self):
        #model
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #base_model = VGG16(weights=weights, include_top=False, input_shape=self.input_shape)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # temporal stream (frozen)
        if self.saved_temporal_weights is None:
            print("[ERROR] No saved_temporal_weights weights file!")
        else:
            model.load_weights(self.saved_temporal_weights)
        for layer in model.layers:
            layer.trainable = False

        return model

    # CNN model for the pose stream
    def cnn_pose(self):
        #model
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model = VGG16(weights=weights, include_top=False, input_shape=self.input_shape)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # pose stream (frozen)
        if self.saved_pose_weights is None:
            print("[ERROR] No saved_pose_weights weights file!")
        else:
            model.load_weights(self.saved_pose_weights)
        for layer in model.layers:
            layer.trainable = False

        return model
