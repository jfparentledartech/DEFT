from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .datasets.mot import MOT, MOT_prediction
from .datasets.nuscenes import nuScenes, nuScenes_prediction
from .datasets.pixset import PixSet, PixSet_prediction
from .datasets.kitti_tracking import KITTITracking, KITTITracking_prediction
from .datasets.custom_dataset import CustomDataset

dataset_factory = {
    "custom": CustomDataset,
    "mot": MOT,
    "nuscenes": nuScenes,
    "pixset": PixSet,
    "kitti_tracking": KITTITracking,
}

dataset_factory_prediction = {
    "mot": MOT_prediction,
    "nuscenes": nuScenes_prediction,
    "pixset": PixSet_prediction,
    "kitti_tracking": KITTITracking_prediction,
}


def get_dataset(dataset, prediction_model=False):
    if prediction_model:
        return dataset_factory_prediction[dataset]
    else:
        return dataset_factory[dataset]
