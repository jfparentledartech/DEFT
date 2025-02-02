# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Xingyi Zhou (zhouxy@cs.utexas.edu)
# Source: https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/detector.py
# Modified by Mohamed Chaabane
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
import time
import torch
import math
import pandas as pd
from lib.model.model import create_model, load_model
from lib.model.decode import generic_decode
from lib.model.utils import flip_tensor, flip_lr_off, flip_lr
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import draw_umich_gaussian, gaussian_radius
from lib.utils.post_process import generic_post_process
from lib.utils.debugger import Debugger
from lib.utils.tracker import Tracker
from lib.dataset.dataset_factory import get_dataset
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from lib.utils.ddd_utils import nms

NUSCENES_TRACKING_NAMES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
]
nuscenes_class_name = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

PIXSET_TRACKING_NAMES = [
    'car',
    'truck',
    'bus',
    'pedestrian',
    'motorcyclist',
    'cyclist',
    'van'
]

pixset_class_name = [
    'car',
    'truck',
    'bus',
    'pedestrian',
    'motorcyclist',
    'cyclist',
    'van'
]


NMS = True


class Detector(object):
    def __init__(self, opt):
        opt.device = torch.device("cuda") if opt.gpus[0] >= 0 else torch.device("cpu")
        # print("Creating model...")
        self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
        self.model = load_model(self.model, opt.load_model, opt)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.opt = opt
        self.trained_dataset = get_dataset(opt.dataset)
        self.mean = np.array(self.trained_dataset.mean, dtype=np.float32).reshape(
            1, 1, 3
        )
        self.std = np.array(self.trained_dataset.std, dtype=np.float32).reshape(1, 1, 3)
        #     self.pause = not opt.no_pause
        self.rest_focal_length = (
            self.trained_dataset.rest_focal_length
            if self.opt.test_focal_length < 0
            else self.opt.test_focal_length
        )
        self.flip_idx = self.trained_dataset.flip_idx
        self.cnt = 0
        self.pre_images = None
        self.pre_traces = None
        self.pre_image_ori = None
        self.dataset = opt.dataset
        if self.dataset == "nuscenes":
            self.tracker = {}
            for class_name in NUSCENES_TRACKING_NAMES:
                self.tracker[class_name] = Tracker(opt, self.model)
        if self.dataset == "pixset":
            self.tracker = {}
            for class_name in PIXSET_TRACKING_NAMES:
                self.tracker[class_name] = Tracker(opt, self.model)
        if self.dataset not in ["nuscenes", "pixset"]:
            self.tracker = Tracker(opt, self.model)

        self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)
        self.img_height = 100
        self.img_width = 100

    def run(self, image_or_path_or_tensor, meta={}, image_info=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
        self.debugger.clear()
        start_time = time.time()

        # read image
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(""):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor["image"][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        if self.opt.use_pixell:
            trace = np.load(image_info["waveform_file_name"][0])
        loaded_time = time.time()
        load_time += loaded_time - start_time

        detections = []

        # for multi-scale testing
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            if not pre_processed:
                # not prefetch testing or demo
                images, meta = self.pre_process(image, scale, meta)
            else:
                # prefetch testing
                images = pre_processed_images["images"][scale][0]
                meta = pre_processed_images["meta"][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
                if "pre_detections" in pre_processed_images["meta"]:
                    meta["pre_detections"] = pre_processed_images["meta"]["pre_detections"]
                if "cur_detections" in pre_processed_images["meta"]:
                    meta["cur_detections"] = pre_processed_images["meta"]["cur_detections"]

            images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)
            traces = None
            if self.opt.use_pixell:
                traces = torch.Tensor(trace.transpose(2,0,1)).reshape((1,256,8,49)).to(self.opt.device, non_blocking=self.opt.non_block_test)

            # initializing tracker
            pre_hms, pre_inds = None, None

            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # run the network
            # output: the output feature maps, only used for visualizing
            # detections: output tensors after extracting peaks
            output, detections_peaks, forward_time, FeatureMaps = self.process(
                images, traces, self.pre_images, self.pre_traces, pre_hms, pre_inds, return_time=True
            )
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            # convert the cropped and 4x downsampled output coordinate system
            # back to the input image coordinate system
            meta['distortion_coefficients'] = np.asarray(image_info["distortion_coefficients"])

            result = self.post_process(detections_peaks, meta, scale)
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            # print("DETECT0", detections)

            detections.append(result)
            if self.opt.debug >= 2:
                self.debug(
                    self.debugger,
                    images,
                    result,
                    output,
                    scale,
                    pre_images=self.pre_images if not self.opt.no_pre_img else None,
                    pre_hms=pre_hms,
                )

        # merge multi-scale testing results
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time

        # public detection mode in MOT challenge
        if self.opt.public_det:
            results = (
                pre_processed_images["meta"]["cur_detections"]
                if self.opt.public_det
                else None
            )

        if self.dataset == "nuscenes":
            trans_matrix = np.array(image_info["trans_matrix"], np.float32)

            results_by_class = {}
            ddd_boxes_by_class = {}
            depths_by_class = {}
            ddd_boxes_by_class2 = {}
            ddd_org_boxes_by_class = {}
            ddd_box_submission1 = {}
            ddd_box_submission2 = {}
            for class_name in NUSCENES_TRACKING_NAMES:
                results_by_class[class_name] = []
                ddd_boxes_by_class2[class_name] = []
                ddd_boxes_by_class[class_name] = []
                depths_by_class[class_name] = []
                ddd_org_boxes_by_class[class_name] = []
                ddd_box_submission1[class_name] = []
                ddd_box_submission2[class_name] = []
            for det in results:
                cls_id = int(det["class"])
                class_name = nuscenes_class_name[cls_id - 1]
                if class_name not in NUSCENES_TRACKING_NAMES:
                    continue

                if det["score"] < 0.3:
                    continue
                if class_name == "pedestrian" and det["score"] < 0.35:
                    continue
                results_by_class[class_name].append(
                    det["bbox"].tolist() + [det["score"]]
                )
                size = [
                    float(det["dim"][1]),
                    float(det["dim"][2]),
                    float(det["dim"][0]),
                ]
                rot_cam = Quaternion(axis=[0, 1, 0], angle=det["rot_y"])
                translation_submission1 = np.dot(
                    trans_matrix,
                    np.array(
                        [det["loc"][0], det["loc"][1] - size[2], det["loc"][2], 1],
                        np.float32,
                    ),
                ).copy()

                loc = np.array(
                    [det["loc"][0], det["loc"][1], det["loc"][2]], np.float32
                )
                depths_by_class[class_name].append([float(det["loc"][2])].copy())
                trans = [det["loc"][0], det["loc"][1], det["loc"][2]]

                ddd_org_boxes_by_class[class_name].append(
                    [float(det["dim"][0]), float(det["dim"][1]), float(det["dim"][2])]
                    + trans
                    + [det["rot_y"]]
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

                ddd_box_submission1[class_name].append(
                    [
                        float(translation_submission1[0]),
                        float(translation_submission1[1]),
                        float(translation_submission1[2]),
                    ].copy()
                    + size.copy()
                    + rotation.copy()
                )

                q = Quaternion(rotation)
                angle = q.angle if q.axis[2] > 0 else -q.angle

                ddd_boxes_by_class[class_name].append(
                    [
                        size[2],
                        size[0],
                        size[1],
                        box.center[0],
                        box.center[1],
                        box.center[2],
                        angle,
                    ].copy()
                )

            online_targets = []
            for class_name in NUSCENES_TRACKING_NAMES:
                if len(results_by_class[class_name]) > 0 and NMS:
                    boxess = torch.from_numpy(
                        np.array(results_by_class[class_name])[:, :4]
                    )
                    scoress = torch.from_numpy(
                        np.array(results_by_class[class_name])[:, -1]
                    )
                    if class_name == "bus" or class_name == "truck":
                        ovrlp = 0.7
                    else:
                        ovrlp = 0.8
                    keep, count = nms(boxess, scoress, overlap=ovrlp)

                    keep = keep.data.numpy().tolist()
                    keep = sorted(set(keep))
                    results_by_class[class_name] = np.array(
                        results_by_class[class_name]
                    )[keep]

                    ddd_boxes_by_class[class_name] = np.array(
                        ddd_boxes_by_class[class_name]
                    )[keep]
                    depths_by_class[class_name] = np.array(depths_by_class[class_name])[
                        keep
                    ]
                    ddd_org_boxes_by_class[class_name] = np.array(
                        ddd_org_boxes_by_class[class_name]
                    )[keep]
                    ddd_box_submission1[class_name] = np.array(
                        ddd_box_submission1[class_name]
                    )[keep]

                online_targets += self.tracker[class_name].update(
                    results_by_class[class_name],
                    FeatureMaps,
                    ddd_boxes=ddd_boxes_by_class[class_name],
                    depths_by_class=depths_by_class[class_name],
                    ddd_org_boxes=ddd_org_boxes_by_class[class_name],
                    submission=ddd_box_submission1[class_name],
                    classe=class_name,
                )

        if self.dataset == "pixset":
            trans_matrix = np.eye(4)
            results_by_class = {}
            ddd_boxes_by_class = {}
            depths_by_class = {}
            ddd_boxes_by_class2 = {}
            ddd_org_boxes_by_class = {}
            ddd_box_submission1 = {}
            ddd_box_submission2 = {}
            for class_name in PIXSET_TRACKING_NAMES:
                results_by_class[class_name] = []
                ddd_boxes_by_class2[class_name] = []
                ddd_boxes_by_class[class_name] = []
                depths_by_class[class_name] = []
                ddd_org_boxes_by_class[class_name] = []
                ddd_box_submission1[class_name] = []
                ddd_box_submission2[class_name] = []
            for det in results:
                cls_id = int(det["class"])
                class_name = pixset_class_name[cls_id - 1]
                if class_name not in PIXSET_TRACKING_NAMES:
                    continue

                if det["score"] < 0.3:
                    continue
                if class_name == "pedestrian" and det["score"] < 0.35:
                    continue
                results_by_class[class_name].append(
                    det["bbox"].tolist() + [det["score"]]
                )
                size = [
                    float(det["dim"][1]),
                    float(det["dim"][2]),
                    float(det["dim"][0]),
                ]
                rot_cam = Quaternion(axis=[0, 1, 0], angle=det["rot_y"])
                translation_submission1 = np.dot(
                    trans_matrix,
                    np.array(
                        [det["loc"][0], det["loc"][1] - size[2], det["loc"][2], 1],
                        np.float32,
                    ),
                ).copy()

                loc = np.array(
                    [det["loc"][0], det["loc"][1], det["loc"][2]], np.float32
                )
                depths_by_class[class_name].append([float(det["loc"][2])].copy())
                trans = [det["loc"][0], det["loc"][1], det["loc"][2]]

                ddd_org_boxes_by_class[class_name].append(
                    [float(det["dim"][0]), float(det["dim"][1]), float(det["dim"][2])]
                    + trans
                    + [det["rot_y"]]
                )

                box = Box(loc, size, rot_cam, name="2", token="1")
                box.translate(np.array([0, -box.wlh[2] / 2, 0]))
                # box.rotate(Quaternion(image_info["cs_record_rot"]))
                # box.translate(np.array(image_info["cs_record_trans"]))
                # box.rotate(Quaternion(image_info["pose_record_rot"]))
                # box.translate(np.array(image_info["pose_record_trans"]))
                rotation = box.orientation
                rotation = [
                    float(rotation.w),
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                ]

                ddd_box_submission1[class_name].append(
                    [
                        float(translation_submission1[0]),
                        float(translation_submission1[1]),
                        float(translation_submission1[2]),
                    ].copy()
                    + size.copy()
                    + rotation.copy()
                )

                q = Quaternion(rotation)
                angle = q.angle if q.axis[2] > 0 else -q.angle

                ddd_boxes_by_class[class_name].append(
                    [
                        size[2],
                        size[0],
                        size[1],
                        box.center[0],
                        box.center[1],
                        box.center[2],
                        # angle, # TODO use det["rot_y"] ?
                        det["rot_y"]
                    ].copy()
                )

            online_targets = []
            for class_name in PIXSET_TRACKING_NAMES:
                if len(results_by_class[class_name]) > 0 and NMS:
                    boxess = torch.from_numpy(
                        np.array(results_by_class[class_name])[:, :4]
                    )
                    scoress = torch.from_numpy(
                        np.array(results_by_class[class_name])[:, -1]
                    )
                    if class_name == "bus" or class_name == "truck":
                        ovrlp = 0.7
                    else:
                        ovrlp = 0.8
                    keep, count = nms(boxess, scoress, overlap=ovrlp)

                    keep = keep.data.numpy().tolist()
                    keep = sorted(set(keep))
                    results_by_class[class_name] = np.array(
                        results_by_class[class_name]
                    )[keep]

                    ddd_boxes_by_class[class_name] = np.array(
                        ddd_boxes_by_class[class_name]
                    )[keep]
                    depths_by_class[class_name] = np.array(depths_by_class[class_name])[
                        keep
                    ]
                    ddd_org_boxes_by_class[class_name] = np.array(
                        ddd_org_boxes_by_class[class_name]
                    )[keep]
                    ddd_box_submission1[class_name] = np.array(
                        ddd_box_submission1[class_name]
                    )[keep]

                online_targets += self.tracker[class_name].update(
                    results_by_class[class_name],
                    FeatureMaps,
                    ddd_boxes=ddd_boxes_by_class[class_name],
                    depths_by_class=depths_by_class[class_name],
                    ddd_org_boxes=ddd_org_boxes_by_class[class_name],
                    submission=ddd_box_submission1[class_name],
                    classe=class_name
                )

        else:

            online_targets = self.tracker.update(results, FeatureMaps)

        return online_targets

    def _transform_scale(self, image, scale=1):
        """
      Prepare input image in different testing modes.
        Currently support: fix short size/ center crop to a fixed size/
        keep original resolution but pad to a multiplication of 32
    """
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_short > 0:
            if height < width:
                inp_height = self.opt.fix_short
                inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
            else:
                inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
                inp_width = self.opt.fix_short
            c = np.array([width / 2, height / 2], dtype=np.float32)
            s = np.array([width, height], dtype=np.float32)
        elif self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2.0, new_height / 2.0], dtype=np.float32)
            s = max(height, width) * 1.0
            # s = np.array([inp_width, inp_height], dtype=np.float32)
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, c, s, inp_width, inp_height, height, width

    def pre_process(self, image, scale, input_meta={}):
        """
    Crop, resize, and normalize image. Gather meta data for post processing
      and tracking.
    """
        resized_image, c, s, inp_width, inp_height, height, width = self._transform_scale(
            image
        )
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        out_height = inp_height // self.opt.down_ratio
        out_width = inp_width // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR
        )
        inp_image = ((inp_image / 255.0 - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {
            "calib": np.array(input_meta["calib"], dtype=np.float32)
            if "calib" in input_meta
            else self._get_default_calib(width, height)
        }
        meta.update(
            {
                "c": c,
                "s": s,
                "height": height,
                "width": width,
                "out_height": out_height,
                "out_width": out_width,
                "inp_height": inp_height,
                "inp_width": inp_width,
                "trans_input": trans_input,
                "trans_output": trans_output,
            }
        )
        if "pre_detections" in input_meta:
            meta["pre_detections"] = input_meta["pre_detections"]
        if "cur_detections" in input_meta:
            meta["cur_detections"] = input_meta["cur_detections"]
        return images, meta

    def _trans_bbox(self, bbox, trans, width, height):
        """
    Transform bounding boxes according to image crop.
    """
        bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        return bbox

    def _get_additional_inputs(self, detections, meta, with_hm=True):
        """
    Render input heatmap from previous trackings.
    """
        trans_input, trans_output = meta["trans_input"], meta["trans_output"]
        inp_width, inp_height = meta["inp_width"], meta["inp_height"]
        out_width, out_height = meta["out_width"], meta["out_height"]
        input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

        output_inds = []
        for det in detections:
            if det["score"] < self.opt.pre_thresh or det["active"] == 0:
                continue
            bbox = self._trans_bbox(det["bbox"], trans_input, inp_width, inp_height)
            bbox_out = self._trans_bbox(
                det["bbox"], trans_output, out_width, out_height
            )
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct_int = ct.astype(np.int32)
                if with_hm:
                    draw_umich_gaussian(input_hm[0], ct_int, radius)
                ct_out = np.array(
                    [(bbox_out[0] + bbox_out[2]) / 2, (bbox_out[1] + bbox_out[3]) / 2],
                    dtype=np.int32,
                )
                output_inds.append(ct_out[1] * out_width + ct_out[0])
        if with_hm:
            input_hm = input_hm[np.newaxis]
            if self.opt.flip_test:
                input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
            input_hm = torch.from_numpy(input_hm).to(self.opt.device)
        output_inds = np.array(output_inds, np.int64).reshape(1, -1)
        output_inds = torch.from_numpy(output_inds).to(self.opt.device)
        return input_hm, output_inds

    def _get_default_calib(self, width, height):
        calib = np.array(
            [
                [self.rest_focal_length, 0, width / 2, 0],
                [0, self.rest_focal_length, height / 2, 0],
                [0, 0, 1, 0],
            ]
        )
        return calib

    def _sigmoid_output(self, output):
        if "hm" in output:
            output["hm"] = output["hm"].sigmoid_()
        if "hm_hp" in output:
            output["hm_hp"] = output["hm_hp"].sigmoid_()
        if "dep" in output:
            output["dep"] = 1.0 / (output["dep"].sigmoid() + 1e-6) - 1.0
            output["dep"] *= self.opt.depth_scale
        return output

    def _flip_output(self, output):
        average_flips = ["hm", "wh", "dep", "dim"]
        neg_average_flips = ["amodel_offset"]
        single_flips = [
            "ltrb",
            "nuscenes_att",
            "velocity",
            "ltrb_amodal",
            "reg",
            "hp_offset",
            "rot",
            "tracking",
            "pre_hm",
        ]
        for head in output:
            if head in average_flips:
                output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
            if head in neg_average_flips:
                flipped_tensor = flip_tensor(output[head][1:2])
                flipped_tensor[:, 0::2] *= -1
                output[head] = (output[head][0:1] + flipped_tensor) / 2
            if head in single_flips:
                output[head] = output[head][0:1]
            if head == "hps":
                output["hps"] = (
                    output["hps"][0:1] + flip_lr_off(output["hps"][1:2], self.flip_idx)
                ) / 2
            if head == "hm_hp":
                output["hm_hp"] = (
                    output["hm_hp"][0:1] + flip_lr(output["hm_hp"][1:2], self.flip_idx)
                ) / 2

        return output

    def process(
        self, images, traces, pre_images=None, pre_traces=None, pre_hms=None, pre_inds=None, return_time=False
    ):
        with torch.no_grad():
            torch.cuda.synchronize()
            output, FeatureMaps = self.model(images, traces, pre_images, pre_traces, pre_hms)
            output = output[-1]
            output = self._sigmoid_output(output)
            output.update({"pre_inds": pre_inds})
            if self.opt.flip_test:
                output = self._flip_output(output)
            torch.cuda.synchronize()
            forward_time = time.time()

            detections = generic_decode(output, K=self.opt.K, opt=self.opt)
            torch.cuda.synchronize()
            for k in detections:
                detections[k] = detections[k].detach().cpu().numpy()
        if return_time:
            return output, detections, forward_time, FeatureMaps
        else:
            return output, detections, FeatureMaps

    def post_process(self, detections, meta, scale=1):
        detections = generic_post_process(
            self.opt,
            detections,
            [meta["c"]],
            [meta["s"]],
            meta["out_height"],
            meta["out_width"],
            self.opt.num_classes,
            [meta["calib"]],
            meta["height"],
            meta["width"],
            [meta["distortion_coefficients"]],
        )
        self.this_calib = meta["calib"]

        if scale != 1:
            for i in range(len(detections[0])):
                for k in ["bbox", "hps"]:
                    if k in detections[0][i]:
                        detections[0][i][k] = (
                            np.array(detections[0][i][k], np.float32) / scale
                        ).tolist()
        return detections[0]

    def merge_outputs(self, detections):
        assert len(self.opt.test_scales) == 1, "multi_scale not supported!"
        results = []
        for i in range(len(detections[0])):
            if detections[0][i]["score"] > self.opt.out_thresh:
                results.append(detections[0][i])
        return results

    def debug(
        self, debugger, images, detections, output, scale=1, pre_images=None, pre_hms=None
    ):
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((img * self.std + self.mean) * 255.0), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output["hm"][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, "pred_hm")
        if "hm_hp" in output:
            pred = debugger.gen_colormap_hp(output["hm_hp"][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, "pred_hmhp")

        if pre_images is not None:
            pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            pre_img = np.clip(
                ((pre_img * self.std + self.mean) * 255.0), 0, 255
            ).astype(np.uint8)
            debugger.add_img(pre_img, "pre_img")
            if pre_hms is not None:
                pre_hm = debugger.gen_colormap(pre_hms[0].detach().cpu().numpy())
                debugger.add_blend_img(pre_img, pre_hm, "pre_hm")

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id="generic")
        if self.opt.tracking:
            debugger.add_img(
                self.pre_image_ori if self.pre_image_ori is not None else image,
                img_id="previous",
            )
            self.pre_image_ori = image

        for j in range(len(results)):
            if results[j]["score"] > self.opt.vis_thresh:
                if "active" in results[j] and results[j]["active"] == 0:
                    continue
                item = results[j]
                if "bbox" in item:
                    sc = (
                        item["score"]
                        if self.opt.demo == "" or not ("tracking_id" in item)
                        else item["tracking_id"]
                    )
                    sc = item["tracking_id"] if self.opt.show_track_color else sc

                    debugger.add_coco_bbox(
                        item["bbox"], item["class"] - 1, sc, img_id="generic"
                    )

                if "tracking" in item:
                    debugger.add_arrow(item["ct"], item["tracking"], img_id="generic")

                tracking_id = item["tracking_id"] if "tracking_id" in item else -1
                if (
                    "tracking_id" in item
                    and self.opt.demo == ""
                    and not self.opt.show_track_color
                ):
                    debugger.add_tracking_id(
                        item["ct"], item["tracking_id"], img_id="generic"
                    )

                if (item["class"] in [1, 2]) and "hps" in item:
                    debugger.add_coco_hp(
                        item["hps"], tracking_id=tracking_id, img_id="generic"
                    )

        if (
            len(results) > 0
            and "dep" in results[0]
            and "alpha" in results[0]
            and "dim" in results[0]
        ):
            debugger.add_3d_detection(
                image
                if not self.opt.qualitative
                else cv2.resize(
                    debugger.imgs["pred_hm"], (image.shape[1], image.shape[0])
                ),
                False,
                results,
                self.this_calib,
                vis_thresh=self.opt.vis_thresh,
                img_id="ddd_pred",
            )
            debugger.add_bird_view(
                results,
                vis_thresh=self.opt.vis_thresh,
                img_id="bird_pred",
                cnt=self.cnt,
            )
            if self.opt.show_track_color and self.opt.debug == 4:
                del debugger.imgs["generic"], debugger.imgs["bird_pred"]

    def reset_tracking(self, opt):
        if self.dataset == "nuscenes":
            self.tracker = {}
            for class_name in NUSCENES_TRACKING_NAMES:
                self.tracker[class_name] = Tracker(
                    opt, self.model, h=self.img_height, w=self.img_width
                )
        if self.dataset == "pixset":
            self.tracker = {}
            for class_name in PIXSET_TRACKING_NAMES:
                self.tracker[class_name] = Tracker(
                    opt, self.model, h=self.img_height, w=self.img_width
                )
        else:
            self.tracker = Tracker(opt, self.model, h=self.img_height, w=self.img_width)
        self.pre_images = None
        self.pre_image_ori = None

    def update_public_detections(self, detections_file):

        self.det_file = pd.read_csv(detections_file, header=None, sep=" ")
        self.det_group = self.det_file.groupby(0)
        self.det_group_keys = self.det_group.indices.keys()
