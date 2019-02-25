"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from fuse_train_model import ResearchModels
from fuse_train_data import DataSet, get_generators
import time
import os.path
from os import makedirs

def fixed_schedule(epoch):
    initial_lr = (1.e-4) * 2
    lr = initial_lr


    lr = lr * ((0.70)**epoch)

    return lr

def train(saved_model=None,
            class_limit=None,
            saved_spatial_weights=None,
            saved_temporal_weights=None,
            saved_pose_weights=None,
            image_shape=(224, 224),
            original_image_shape=(341, 256),
            batch_size=32,
            fuse_method='average',
            nb_epoch=100, name_str=None):

    # Get local time.
    time_str = time.strftime("%y%m%d%H%M", time.localtime())

    if name_str == None:
        name_str = time_str

    # Callbacks: Save the model.
    directory1 = os.path.join('out', 'checkpoints', name_str)
    if not os.path.exists(directory1):
            os.makedirs(directory1)
    checkpointer = ModelCheckpoint(
            filepath=os.path.join(directory1,
                    'three_stream_avrage-{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            # save_best_only=True
            monitor='val_acc')

    # Callbacks: TensorBoard
    directory2 = os.path.join('out', 'TB', name_str)
    if not os.path.exists(directory2):
            os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stopper.
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join('out', 'logs', name_str)
    if not os.path.exists(directory3):
            os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
            str(timestamp) + '.log'))

    # Learning rate schedule.
    lr_schedule = LearningRateScheduler(fixed_schedule, verbose=0)

    print("class_limit = ", class_limit)
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            class_limit=class_limit,
            batch_size=batch_size
                )
    else:
        data = DataSet(
            class_limit=class_limit,
            image_shape=image_shape,
            batch_size=batch_size
                )


    # Get generators.
    generator, val_generator = data.get_generators()
    train_steps = data.n_batch_train
    valid_steps = data.n_batch_test

    # Get the model.
    three_stream_fuse = ResearchModels(nb_classes=len(data.classes), image_shape=image_shape, saved_model=saved_model,
                                       saved_temporal_weights=saved_temporal_weights,
                                       saved_spatial_weights=saved_spatial_weights,
                                       saved_pose_weights=saved_pose_weights,
                                       fuse_method=fuse_method)

    #three_stream_fuse.freeze_all_but_top()
#
    ## Fit!
    ##Use fit generator.
    #three_stream_fuse.model.fit_generator(
    #            generator=generator,
    #            steps_per_epoch=train_steps,
    #            epochs=10,
    #            verbose=1,
    #            callbacks=[tb, early_stopper, csv_logger, checkpointer],
    #            validation_data=val_generator,
    #            validation_steps=valid_steps,
    #            max_queue_size=20,
    #            workers=4,
    #            use_multiprocessing=False)
#
#
    #three_stream_fuse.unfreeze_all()
    #Fit!
    # Use fit generator.
    three_stream_fuse.model.fit_generator(
                generator=generator,
                steps_per_epoch=train_steps,
                epochs=nb_epoch,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger, checkpointer],
                validation_data=val_generator,
                validation_steps=valid_steps,
                max_queue_size=20,
                workers=4,
                use_multiprocessing=False)


def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_spatial_weights = None #".\out\checkpoints\\willow\\spatial\\resnet50.hdf5"
    saved_temporal_weights = None #".\out\checkpoints\\willow\\temporal\\resnet50.hdf5"
    saved_pose_weights = None #".\out\checkpoints\\willow\\pose\\resnet50.hdf5"
    saved_model = None #".\out\checkpoints\willow\\fuse concat\\005-0.530.hdf5"
    class_limit = None
    image_shape=(224, 224)
    original_image_shape=(341, 256)
    batch_size = 8
    fuse_method = 'average'
    "=============================================================================="

    train(
            saved_model = saved_model,
            class_limit=class_limit,
            saved_spatial_weights=saved_spatial_weights,
            saved_temporal_weights=saved_temporal_weights,
            saved_pose_weights=saved_pose_weights,
            image_shape=image_shape,
            original_image_shape=original_image_shape,
            batch_size=batch_size,
            fuse_method=fuse_method
            )

if __name__ == '__main__':
    main()
