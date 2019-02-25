"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from fuse_validate_model import ResearchModels
from fuse_validate_data import DataSet, get_generators
import time
import os.path
from os import makedirs

def test_1epoch_fuse(
            class_limit=None,
            saved_model=None,
            saved_spatial_weights=None,
            saved_temporal_weights=None,
            saved_pose_weights=None,
            image_shape=(224, 224),
            original_image_shape=(341, 256),
            batch_size=32,
            fuse_method='average'):

    print("class_limit = ", class_limit)

    # Get the data.
    data = DataSet(
            class_limit=class_limit,
            image_shape=image_shape,
            original_image_shape=original_image_shape,
            batch_size=batch_size
            )

    val_generator = data.get_generators() # Get the validation generator
    steps = data.n_batch


    # Get the model.
    three_stream_fuse = ResearchModels(nb_classes=len(data.classes), image_shape=image_shape, saved_model=saved_model,
                                       saved_temporal_weights=saved_temporal_weights,
                                       saved_spatial_weights=saved_spatial_weights,
                                       saved_pose_weights=saved_pose_weights,
                                       fuse_method=fuse_method)


    # Evaluate!
    three_stream_fuse.model.fit_generator(generator=val_generator, steps_per_epoch=steps, max_queue_size=1, shuffle=False)

def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_spatial_weights = ".\out\checkpoints\\willow\\spatial\\resnet50.hdf5"
    saved_temporal_weights = ".\out\checkpoints\\willow\\temporal\\resnet50.hdf5"
    saved_pose_weights = ".\out\checkpoints\\willow\\pose\\resnet50.hdf5"
    saved_model = ".\out\checkpoints\willow\\fuse concat\\005-0.530.hdf5"
    class_limit = None
    image_shape=(224, 224)
    original_image_shape=(341, 256)
    batch_size = 32
    fuse_method = 'average'
    "=============================================================================="

    test_1epoch_fuse(
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
