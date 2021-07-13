from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import copy
import pickle
import motmetrics as mm
import argparse

import _init_paths
from lib.opts import opts
from lib.logger import Logger
from lib.utils.utils import AverageMeter
from lib.dataset.dataset_factory import dataset_factory

from lib.utils.pixset_metrics import compute_metrics
from lib.detector import Detector
import json

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

min_box_area = 20


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.get_ann_ids = dataset.coco.getAnnIds
        self.load_annotations = dataset.coco.loadAnns
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.get_default_calib = dataset.get_default_calib
        self.opt = opt

    def __getitem__(self, index):
        self.images.sort() # TODO remove
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)

        annotation_ids = self.get_ann_ids(imgIds=[img_id])
        annotations = self.load_annotations(ids=annotation_ids)

        images, meta = {}, {}
        for scale in self.opt.test_scales:
            input_meta = {}
            calib = (
                img_info["calib"]
                if "calib" in img_info
                else self.get_default_calib(image.shape[1], image.shape[0])
            )
            input_meta["calib"] = calib
            images[scale], meta[scale] = self.pre_process_func(image, scale, input_meta)
        ret = {
            "images": images,
            "image": image,
            "meta": meta,
            "frame_id": img_info["frame_id"],
            "annotations": annotations
        }
        if "frame_id" in img_info and img_info["frame_id"] == 1:
            ret["is_first_frame"] = 1
            ret["video_id"] = img_info["video_id"]
        return img_id, ret, img_info

    def __len__(self):
        return len(self.images)


def eval_pixset(opt, epoch):
    if not opt.not_set_cuda_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    Dataset = dataset_factory[opt.test_dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)

    split = "val" if not opt.trainval else "test"
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    if opt.load_results != "":
        load_results = json.load(open(opt.load_results, "r"))
        for img_id in load_results:
            for k in range(len(load_results[img_id])):
                if load_results[img_id][k]["class"] - 1 in opt.ignore_loaded_cats:
                    load_results[img_id][k]["score"] = -1
    else:
        load_results = {}

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters

    vehicle_acc = mm.MOTAccumulator(auto_id=True)
    pedestrian_acc = mm.MOTAccumulator(auto_id=True)

    bar = Bar(f'Computing PixSet Tracking Metrics... {dataset} frames', max=len(data_loader))

    for ind, (img_id, pre_processed_images, img_info) in enumerate(data_loader):
        bar.next()

        if ind >= num_iters:
            break

        if opt.tracking and ("is_first_frame" in pre_processed_images):
            if "{}".format(int(img_id.numpy().astype(np.int32)[0])) in load_results:
                pre_processed_images["meta"]["pre_dets"] = load_results[
                    "{}".format(int(img_id.numpy().astype(np.int32)[0]))
                ]
            else:
                print(
                    "No pre_dets for",
                    int(img_id.numpy().astype(np.int32)[0]),
                    ". Use empty initialization.",
                )
                pre_processed_images["meta"]["pre_dets"] = []

            detector.reset_tracking(opt)

            print("Start tracking video", int(pre_processed_images["video_id"]))

        online_targets = detector.run(pre_processed_images, image_info=img_info)

        vehicle_gt_list, vehicle_hyp_list, vehicle_distances = compute_metrics(pre_processed_images['annotations'], online_targets, category='vehicle')
        pedestrian_gt_list, pedestrian_hyp_list, pedestrian_distances = compute_metrics(pre_processed_images['annotations'], online_targets, category='pedestrian')
        vehicle_acc.update(vehicle_gt_list, vehicle_hyp_list, vehicle_distances)
        pedestrian_acc.update(pedestrian_gt_list, pedestrian_hyp_list, pedestrian_distances)

    mh = mm.metrics.create()
    summary = mh.compute(vehicle_acc, metrics=['num_frames', 'mota', 'motp', 'precision', 'recall'], name=f'epoch {epoch} vehicle')
    print(summary)
    save_summary(summary, f'vehicle')

    mh = mm.metrics.create()
    summary = mh.compute(pedestrian_acc, metrics=['num_frames', 'mota', 'motp', 'precision', 'recall'],  name=f'epoch {epoch} pedestrian')
    print(summary)
    save_summary(summary, f'pedestrian')
    bar.finish()


def save_summary(summary, acc_name):
    with open(f"./pixset_results/{acc_name}.txt", "a") as text_file:
        text_file.write('\n' + summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch")
    args = parser.parse_args()
    epoch = args.epoch
    print(epoch)
    filename = 'train_opt_pixset.txt'

    with open(filename, 'rb') as f:
        opt = pickle.load(f)

    print(f'Using pixell -> ', opt.use_pixell)

    eval_pixset(opt, epoch)
