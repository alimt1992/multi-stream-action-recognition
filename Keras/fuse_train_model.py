import numpy as np
#from keras_resnet101 import resnet101_model as Resnet101
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, load_model, Model
from keras.layers import Input, average, concatenate, GlobalAveragePooling2D, add, Lambda, Average
from keras.layers import TimeDistributed, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class ResearchModels():
    def __init__(self, nb_classes, image_shape = (224, 224), saved_model=None, saved_temporal_weights=None,
                 saved_spatial_weights=None, saved_pose_weights=None, fuse_methood='average'):

        """
        `nb_classes` = the number of classes to predict
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


        if fuse_methood=='chained':
            self.model = self.three_stream_fuse_chained()
        elif fuse_methood=='concatenate':
            self.model = self.three_stream_fuse_concat()
        else:
            self.model = self.three_stream_fuse_average()


        # Load model
        # If saved fuse model exists, directly load
        if self.saved_model is not None:
            print("\nLoading model %s" % self.saved_model)
            self.model.load_weights(self.saved_model)
        # Otherwise build the model and load weights for both streams
        else:

            print("\nLoading the three-stream model...")

        optimizer = Adam(lr=0.0002, decay=0.2)
        #optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

        #self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)


    # Three-stream fused model
    def three_stream_fuse_average(self):
        # spatial stream (frozen)
        cnn_spatial = self.cnn_spatial()

        # temporal stream (frozen)
        cnn_temporal = self.cnn_temporal()

        # temporal stream (frozen)
        cnn_pose = self.cnn_pose()

        # fused by taking average
        #outputs = average([cnn_spatial.output, cnn_temporal.output, cnn_pose.output])

        out1 = Lambda(lambda x: x * 5/8)(cnn_spatial.output)
        out2 = Lambda(lambda x: x * 1/8)(cnn_temporal.output)
        out3 = Lambda(lambda x: x * 2/8)(cnn_pose.output)

        outputs = add([out1, out2, out3])

        model = Model(inputs=[cnn_spatial.input, cnn_temporal.input, cnn_pose.input], outputs=outputs)

        #for layer in model.layers:
        #    layer.trainable = False

        return model

    def three_stream_fuse_concat(self):
        # spatial stream (frozen)
        cnn_spatial = self.cnn_spatial()
        cnn_spatial = Model(inputs=cnn_spatial.input,outputs=cnn_spatial.get_layer("global_average_pooling2d_1_1").output)

        # temporal stream (frozen)
        cnn_temporal = self.cnn_temporal()
        cnn_temporal = Model(inputs=cnn_temporal.input,outputs=cnn_temporal.get_layer("global_average_pooling2d_2_2").output)

        # pose stream (frozen)
        cnn_pose = self.cnn_pose()
        cnn_pose = Model(inputs=cnn_pose.input,outputs=cnn_pose.get_layer("global_average_pooling2d_3_3").output)

        # fused by concatenating
        x = concatenate([cnn_spatial.output, cnn_temporal.output, cnn_pose.output])

        # and a logistic layer
        x = Dense(1024, activation="relu")(x)
        outputs = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x)

        model = Model([cnn_spatial.input, cnn_temporal.input, cnn_pose.input], outputs)

        return model

    def three_stream_fuse_chained(self):
        # spatial stream (frozen)
        cnn_spatial = self.cnn_spatial()
        cnn_spatial = Model(inputs=cnn_spatial.input,outputs=cnn_spatial.get_layer("global_average_pooling2d_1_1").output)

        # temporal stream (frozen)
        cnn_temporal = self.cnn_temporal()
        cnn_temporal = Model(inputs=cnn_temporal.input,outputs=cnn_temporal.get_layer("global_average_pooling2d_2_2").output)

        # pose stream (frozen)
        cnn_pose = self.cnn_pose()
        cnn_pose = Model(inputs=cnn_pose.input,outputs=cnn_pose.get_layer("global_average_pooling2d_3_3").output)

        # fused by taking average
        x1 = cnn_pose.output
        x2 = concatenate([cnn_pose.output, cnn_temporal.output])
        x3 = concatenate([cnn_pose.output, cnn_temporal.output, cnn_spatial.output])

        #x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        x1 = Dense(1024, activation="relu")(x1)
        output1 = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x1)
        x2 = Dense(self.nb_classes, activation="relu")(x2)
        output2 = Dense(1024, activation=(lambda x: K.tf.nn.softmax(x)))(x2)
        x3 = Dense(self.nb_classes, activation="relu")(x3)
        output3 = Dense(1024, activation=(lambda x: K.tf.nn.softmax(x)))(x3)

        model = Model([cnn_spatial.input, cnn_temporal.input, cnn_pose.input], outputs=[output1, output2, output3])

        return model


    # CNN model for the spatial stream
    def cnn_spatial(self):
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape_spatial)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape_spatial)
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape_spatial)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x)
        #predictions = Dense(self.nb_classes, activation="relu")(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # spatial stream (frozen)
        if self.saved_spatial_weights is None:
            print("[ERROR] No saved_spatial_weights weights file!")
        else:
            model.load_weights(self.saved_spatial_weights)
        for layer in model.layers:
            layer.name = layer.name + str("_1")
            layer.trainable = True

        #"mixed10"

        #model = Model(inputs=model.input,outputs=model.get_layer("global_average_pooling2d_1_1").output)

        return model

    # CNN model for the temporal stream
    def cnn_temporal(self):
        #model
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape_temporal)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape_temporal)
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape_temporal)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x)
        #predictions = Dense(self.nb_classes, activation="relu")(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # temporal stream (frozen)
        if self.saved_temporal_weights is None:
            print("[ERROR] No saved_temporal_weights weights file!")
        else:
            model.load_weights(self.saved_temporal_weights)
        for layer in model.layers:
            layer.name = layer.name + str("_2")
            layer.trainable = True

        #"mixed10"

        #model = Model(inputs=model.input,outputs=model.get_layer("global_average_pooling2d_2_2").output)

        return model

    # CNN model for the pose stream
    def cnn_pose(self):
        #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape_pose)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape_pose)
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape_pose)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(self.nb_classes, activation=(lambda x: K.tf.nn.softmax(x)))(x)
        #predictions = Dense(self.nb_classes, activation="relu")(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # pose stream (frozen)
        if self.saved_pose_weights is None:
            print("[ERROR] No saved_pose_weights weights file!")
        else:
            model.load_weights(self.saved_pose_weights)
        for layer in model.layers:
            layer.name = layer.name + str("_3")
            layer.trainable = True

        #"mixed10"

        #model = Model(inputs=model.input,outputs=model.get_layer("global_average_pooling2d_3_3").output)

        return model

    def freeze_all_but_top(self):
        """Used to train just the top layers of the model."""
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers:
            layer.trainable = False

        self.model.get_layer("dense_1_1").trainable = True
        self.model.get_layer("dense_2_2").trainable = True
        self.model.get_layer("dense_2_2").trainable = True
        self.model.get_layer("concatenate_1").trainable = True
        self.model.get_layer("dense_4").trainable = True
        # compile the model (should be done *after* setting layers to non-trainable)
        optimizer = Adam(lr=0.0002, decay=0.1, amsgrad=True)
        self.model.compile(
            #optimizer='rmsprop',
            optimizer=optimizer,
            loss='categorical_crossentropy', metrics=['accuracy'])

    def unfreeze_all(self):
        """Used to train just the top layers of the model."""
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers:
            layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        #optimizer = SGD(lr=1e-3, momentum=0.9, decay=0.1, nesterov=True)
        optimizer = Adam(lr=0.0002, decay=0.2)
        self.model.compile(
            #optimizer='rmsprop',
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy'])


