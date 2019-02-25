"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from temporal_train_model import ResearchModels
from temporal_train_data import DataSet, get_generators
import time
import os.path
from os import makedirs

def fixed_schedule(epoch):
    initial_lr = (1.e-5) * 2
    lr = initial_lr


    lr = lr * ((0.70)**epoch)

    return lr

def train(saved_model=None,
        class_limit=None, image_shape=(224, 224),
        load_to_memory=False, batch_size=32, nb_epoch=100, name_str=None):

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
                    '{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True,
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
    #lr_schedule = LearningRateScheduler(fixed_schedule, verbose=0)

    print("class_limit = ", class_limit)
    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
                class_limit=class_limit,
                batch_size=batch_size
                )
    else:
        data = DataSet(
                image_shape=image_shape,
                class_limit=class_limit,
                batch_size=batch_size
                )



    # Get generators.
    generator, val_generator = data.get_generators()


    # Get the model.
    temporal_cnn = ResearchModels(nb_classes=len(data.classes), image_shape=image_shape, saved_model=saved_model)


    # Fit!
    if load_to_memory:
        # Use standard fit.
        temporal_cnn.model.fit(
                X,
                y,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=nb_epoch)
    else:
        # Use fit generator.
        temporal_cnn.model.fit_generator(
                generator=generator,
                #steps_per_epoch=steps_per_epoch,
                epochs=nb_epoch,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger, checkpointer],
                validation_data=val_generator,
                #validation_steps=1,
                max_queue_size=20,
                workers=4,
                use_multiprocessing=False)


def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_model = None
    class_limit = None  # int, can be 1-101 or None
    image_shape=(224, 224)
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 500
    name_str = None
    "=============================================================================="

    train(saved_model=saved_model,
            class_limit=class_limit, image_shape=image_shape,
            load_to_memory=load_to_memory, batch_size=batch_size,
            nb_epoch=nb_epoch, name_str=name_str)

if __name__ == '__main__':
    main()
