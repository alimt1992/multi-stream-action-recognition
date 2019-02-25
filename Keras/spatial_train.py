"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from spatial_train_model import ResearchModels
from spatial_train_data import DataSet
import time
import os.path
from os import makedirs

def fixed_schedule(epoch):
    initial_lr = (1.e-4) * 2
    lr = initial_lr


    lr = lr * ((0.9)**(epoch/2))

    return lr

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        #steps_per_epoch=100,
        validation_data=validation_generator,
        #validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks,
        workers=4)
    return model

def train(saved_weights=None,
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
            filepath=os.path.join(directory1, '{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)
            #monitor='val_acc')

    # Callbacks: TensorBoard
    directory2 = os.path.join('out', 'TB', name_str)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stoper
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join('out', 'logs', name_str)
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
        str(timestamp) + '.log'))

    print("class_limit = ", class_limit)

    #lr_schedule = LearningRateScheduler(fixed_schedule, verbose=0)

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
    generators = data.get_generators()

    # Get the model.
    spatial_cnn = ResearchModels(nb_classes=len(data.classes), image_shape=image_shape, saved_weights=saved_weights)

    #print("Get and train the top layers...")
    #model = freeze_all_but_top(model)
    #model = train_model(model, 10, generators)



    #print("Get and train the mid layers...")
    #model = freeze_all_but_mid_and_top(model)
    #model = unfreeze_all(model)
    model = train_model(spatial_cnn.model, nb_epoch, generators, [tb, early_stopper, csv_logger, checkpointer])

def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_weights = None
    #out\checkpoints\\1901072214\\100-1.202.hdf5
    #".\out\checkpoints\\1901072240\\100-0.820.hdf5"
    class_limit = None  # int, can be 1-101 or None
    image_shape=(224, 224)
    load_to_memory = False  # pre-load the sequencea in,o memory
    batch_size = 16
    nb_epoch = 100
    name_str = None
    "=============================================================================="

    train(saved_weights=saved_weights,
            class_limit=class_limit, image_shape=image_shape,
            load_to_memory=load_to_memory, batch_size=batch_size,
            nb_epoch=nb_epoch, name_str=name_str)

if __name__ == '__main__':
    main()
