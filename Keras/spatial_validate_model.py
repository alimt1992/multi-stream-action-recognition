import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Average, GlobalAveragePooling2D
from keras.layers import TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class ResearchModels():
    def __init__(self, nb_classes, image_shape = (224, 224), saved_weights=None):
        """
        `nb_classes` = the number of classes to predict
        `image_shape` = shape of image frame
        `saved_model` = the path to a saved Keras model to load
        """
        self.nb_classes = nb_classes
        self.saved_weights = saved_weights

        self.input_shape = (image_shape[0], image_shape[1], 3)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        self.model = self.cnn_spatial()

        optimizer = Adam()

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)


    # CNN model for the spatial stream
    def cnn_spatial(self, weights='imagenet'):
        # create the base pre-trained model
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        #base_model = VGG16(weights=weights, include_top=False, input_shape=self.input_shape)
    
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x)
    
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        model.load_weights(self.saved_weights)
        for layer in model.layers:
            layer.trainable = False

        return model

