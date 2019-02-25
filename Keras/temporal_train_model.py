import numpy as np
from keras_resnet101 import resnet101_model as Resnet101
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Average, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras import backend as K

class ResearchModels():
    def __init__(self, nb_classes,image_shape = (224, 224), saved_model=None):
        """
        `nb_classes` = the number of classes to predict
        `opt_flow_len` = the length of optical flow frames
        `image_shape` = shape of image frame
        `saved_model` = the path to a saved Keras model to load
        """
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        print("Number of classes:")
        print(self.nb_classes)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.input_shape = (image_shape[0], image_shape[1], 3)
            self.model = self.cnn_temporal()
            self.model.load_weights(self.saved_model)
        else:
            print("Loading CNN model for the temporal stream.")
            self.input_shape = (image_shape[0], image_shape[1], 3)
            self.model = self.cnn_temporal()

        #optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)
        optimizer = Adam(lr=0.0002, decay=0.2)
        #optimizer = Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.2)

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())

    # CNN model for the temporal stream
    def cnn_temporal(self):
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def freeze_all_but_top(self):
        """Used to train just the top layers of the model."""
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers[:-2]:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        #optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)
        optimizer = Adam(lr=0.002, decay=0)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model = self.model
        return model

    def unfreeze_all(self):
        """Used to train just the top layers of the model."""
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers[:-2]:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        #optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        optimizer = Adam(lr=0.002, decay=0)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model = self.model
        return model

