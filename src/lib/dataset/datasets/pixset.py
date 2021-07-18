# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Xingyi Zhou (zhouxy@cs.utexas.edu)
# Source: https://github.com/xingyizhou/CenterTrack
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyquaternion import Quaternion
import numpy as np
import json
import os

from ..generic_dataset import GenericDataset
from ..trajectory_dataset import TrajectoryDataset


class PixSet(GenericDataset):
    # default_resolution = [448, 800] # TODO modify
    default_resolution = [320, 1440] # TODO modify
    class_name = [
        'car',
        'truck',
        'bus',
        'trailer',
        'pedestrian',
        'motorcyclist',
        'cyclist',
        'van'
    ]

    num_categories = len(class_name)

    cat_ids = {i + 1: i + 1 for i in range(num_categories)}
    focal_length = 1200
    max_objs = 128
    _tracking_ignored_class = []
    _vehicles = ["car", "truck", "bus", "trailer"]
    _cycles = ["motorcycle", "bicycle"]
    _pedestrians = ["pedestrian"]
    attribute_to_id = {
        "": 0,
        "cycle.with_rider": 1,
        "cycle.without_rider": 2,
        "pedestrian.moving": 3,
        "pedestrian.standing": 4,
        "pedestrian.sitting_lying_down": 5,
        "vehicle.moving": 6,
        "vehicle.parked": 7,
        "vehicle.stopped": 8,
    }
    id_to_attribute = {v: k for k, v in attribute_to_id.items()}

    def __init__(self, opt, split):
        self.class_name = [
            'car',
            'truck',
            'bus',
            'trailer',
            'pedestrian',
            'motorcyclist',
            'cyclist',
            'van'
        ]
        # self.class_name = ["pedestrian", "vehicle"]
        split_names = {"train": "train", "val": "val"}
        data_dir = os.path.join(opt.data_dir, "pixset")
        if not split == "test":
            split_name = split_names[split]

            img_dir = os.path.join(
                data_dir, "v1.0-mini" if split_name == "mini" else "v1.0-trainval"
            )
            print("Dataset version", opt.dataset_version)
        if opt.dataset_version == "test" or split == "test":
            ann_path = os.path.join(data_dir, "annotations", "test.json")
            img_dir = os.path.join(data_dir, "v1.0-test")
        else:
            ann_path = os.path.join(data_dir, "annotations", "{}{}.json").format(
                opt.dataset_version, split_name
            )

        self.images = None
        super(PixSet, self).__init__(opt, split, ann_path, img_dir)

        self.alpha_in_degree = False
        self.num_samples = len(self.images)

        print("Loaded {} {} samples".format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

     # TODO?
    def convert_coco_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    category_id = item["class"]
                    bbox = item["bbox"]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(item["score"])),
                    }
                    detections.append(detection)
        return detections

    # TODO
    def convert_eval_format(self, results):
        from nuscenes.utils.data_classes import Box

        ret = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": {},
        }
        print("Converting pixset format...")
        for image_id in self.images:
            if not (image_id in results):
                continue
            image_info = self.coco.loadImgs(ids=[image_id])[0]
            sample_token = image_info["sample_token"]
            trans_matrix = np.array(image_info["trans_matrix"], np.float32)
            sensor_id = image_info["sensor_id"]
            sample_results = []
            for item in results[image_id]:
                class_name = (
                    self.class_name[int(item["class"] - 1)]
                    if not ("detection_name" in item)
                    else item["detection_name"]
                )
                if self.opt.tracking and class_name in self._tracking_ignored_class:
                    continue
                score = (
                    float(item["score"])
                    if not ("detection_score" in item)
                    else item["detection_score"]
                )
                if "size" in item:
                    size = item["size"]
                else:
                    size = [
                        float(item["dim"][1]),
                        float(item["dim"][2]),
                        float(item["dim"][0]),
                    ]
                if "translation" in item:
                    translation = item["translation"]
                else:
                    translation = np.dot(
                        trans_matrix,
                        np.array(
                            [
                                item["loc"][0],
                                item["loc"][1] - size[2],
                                item["loc"][2],
                                1,
                            ],
                            np.float32,
                        ),
                    )

                det_id = item["det_id"] if "det_id" in item else -1
                tracking_id = item["tracking_id"] if "tracking_id" in item else 1

                if not ("rotation" in item):
                    rot_cam = Quaternion(axis=[0, 1, 0], angle=item["rot_y"])
                    loc = np.array(
                        [item["loc"][0], item["loc"][1], item["loc"][2]], np.float32
                    )
                    box = Box(loc, size, rot_cam, name="2", token="1")
                    box.translate(np.array([0, -box.wlh[2] / 2, 0]))
                    box.rotate(Quaternion(image_info["cs_record_rot"]))
                    box.translate(np.array(image_info["cs_record_trans"]))
                    box.rotate(Quaternion(image_info["pose_record_rot"]))
                    box.translate(np.array(image_info["pose_record_trans"]))
                    rotation = box.orientation
                    rotation = [
                        float(rotation.w),
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                    ]
                else:
                    rotation = item["rotation"]

                nuscenes_att = (
                    np.array(item["nuscenes_att"], np.float32)
                    if "nuscenes_att" in item
                    else np.zeros(8, np.float32)
                )
                att = ""
                if class_name in self._cycles:
                    att = self.id_to_attribute[np.argmax(nuscenes_att[0:2]) + 1]
                elif class_name in self._pedestrians:
                    att = self.id_to_attribute[np.argmax(nuscenes_att[2:5]) + 3]
                elif class_name in self._vehicles:
                    att = self.id_to_attribute[np.argmax(nuscenes_att[5:8]) + 6]
                if "velocity" in item and len(item["velocity"]) == 2:
                    velocity = item["velocity"]
                else:
                    velocity = item["velocity"] if "velocity" in item else [0, 0, 0]
                    velocity = np.dot(
                        trans_matrix,
                        np.array(
                            [velocity[0], velocity[1], velocity[2], 0], np.float32
                        ),
                    )
                    velocity = [float(velocity[0]), float(velocity[1])]
                result = {
                    "sample_token": sample_token,
                    "translation": [
                        float(translation[0]),
                        float(translation[1]),
                        float(translation[2]),
                    ],
                    "size": size,
                    "rotation": rotation,
                    "velocity": velocity,
                    "detection_name": class_name,
                    "attribute_name": att
                    if not ("attribute_name" in item)
                    else item["attribute_name"],
                    "detection_score": score,
                    "tracking_name": class_name,
                    "tracking_score": score,
                    "tracking_id": tracking_id,
                    "sensor_id": sensor_id,
                    "det_id": det_id,
                }

                sample_results.append(result)
            if sample_token in ret["results"]:
                ret["results"][sample_token] = (
                    ret["results"][sample_token] + sample_results
                )
            else:
                ret["results"][sample_token] = sample_results

        for sample_token in ret["results"].keys():
            confs = sorted(
                [
                    (-d["detection_score"], ind)
                    for ind, d in enumerate(ret["results"][sample_token])
                ]
            )
            ret["results"][sample_token] = [
                ret["results"][sample_token][ind]
                for _, ind in confs[: min(500, len(confs))]
            ]

        return ret

    def save_results(self, results, save_dir, task):
        json.dump(
            self.convert_eval_format(results),
            open("{}/results_pixset_{}.json".format(save_dir, task), "w"),
        )

    def run_eval(self, results, save_dir, epoch):
        task = "tracking" if self.opt.tracking else "det"
        self.save_results(results, save_dir, task)

        os.system(
            "python tools/eval_pixset.py "
            + f" --epoch {epoch}"
        )


class PixSet_prediction(TrajectoryDataset):
    # default_resolution = [448, 800]
    default_resolution = [320, 1440]
    class_name = [
        'car',
        'truck',
        'bus',
        'trailer',
        'pedestrian',
        'motorcyclist',
        'cyclist',
        'van'
    ]

    num_categories = len(class_name)

    cat_ids = {i + 1: i + 1 for i in range(num_categories)}
    focal_length = 1200
    max_objs = 128
    _tracking_ignored_class = []
    # _tracking_ignored_class = ["traffic_cone", "stop sign", "traffic light", "traffic sign", "fire hydrant", "barrier"]
    _vehicles = ["car", "truck", "bus", "trailer", "van", "construction_vehicle", "unclassified vehicle"]
    _cycles = ["motorcyclist", "cyclist"]
    _pedestrians = ["pedestrian"]
    attribute_to_id = {
        "": 0,
        "cycle.with_rider": 1,
        "cycle.without_rider": 2,
        "pedestrian.moving": 3,
        "pedestrian.standing": 4,
        "pedestrian.sitting_lying_down": 5,
        "vehicle.moving": 6,
        "vehicle.parked": 7,
        "vehicle.stopped": 8,
    }
    id_to_attribute = {v: k for k, v in attribute_to_id.items()}

    def __init__(self, opt, split):
        split_names = {"train": "train", "val": "val"}
        data_dir = os.path.join(opt.data_dir, "pixset")
        if not split == "test":
            split_name = split_names[split]

            img_dir = os.path.join(
                data_dir, "v1.0-mini" if split_name == "mini" else "v1.0-trainval"
            )
            print("Dataset version", opt.dataset_version)
        if opt.dataset_version == "test" or split == "test":
            ann_path = os.path.join(data_dir, "annotations", "test.json")
            img_dir = os.path.join(data_dir, "v1.0-test")
        else:
            ann_path = os.path.join(data_dir, "annotations", "{}{}.json").format(
                opt.dataset_version, split_name
            )

        self.images = None
        super(PixSet_prediction, self).__init__(opt, split, ann_path, img_dir)

        self.alpha_in_degree = False
        self.num_samples = len(self.images)

        print("Loaded {} {} samples".format(split, self.num_samples))

    def __len__(self):
        return self.num_samples
