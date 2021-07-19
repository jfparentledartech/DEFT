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
from distutils.util import strtobool

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

pixset_categories = [
    'car',
    'truck',
    'bus',
    'trailer',
    'pedestrian',
    'motorcyclist',
    'cyclist',
    'van'
]

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
        self.images.sort()
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

    # accumulators = [mm.MOTAccumulator(auto_id=True) for _ in pixset_categories]
    accumulators = []
    for _ in pixset_categories:
        accumulator = mm.MOTAccumulator(auto_id=True)
        accumulators.append(accumulator)

    bar = Bar(f'Computing PixSet Tracking Metrics...', max=len(data_loader))

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

        for acc_i in range(len(accumulators)):
            gt_list, hyp_list, distances = compute_metrics(pre_processed_images['annotations'],
                                                           online_targets, eval_type='distance',
                                                           category=pixset_categories[acc_i])
            accumulators[acc_i].update(gt_list, hyp_list, distances)


    for acc_i in range(len(accumulators)):
        mh = mm.metrics.create()
        summary = mh.compute(accumulators[acc_i],
                             metrics=['num_frames', 'mota', 'motp', 'precision', 'recall', 'mostly_tracked',
                                      'partially_tracked', 'mostly_lost'],
                             name=f'{pixset_categories[acc_i]} {epoch}')
        print(summary)
        save_summary(summary, f'{pixset_categories[acc_i]}')
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

    filename = '../options/train_opt_pixset.txt'
    test_filename = '../options/test_opt_pixset.txt'
    # filename = '/home/jfparent/Documents/Stage/DEFT/options/train_opt_pixset.txt'
    # test_filename = '/home/jfparent/Documents/Stage/DEFT/options/test_opt_pixset.txt'

    with open(test_filename, 'rb') as f:
        test_opt = pickle.load(f)

    with open(filename, 'rb') as f:
        opt = pickle.load(f)

    print(f'Using pixell -> ', opt.use_pixell)
    if isinstance(test_opt.lstm, str):
        test_opt.lstm = bool(strtobool(test_opt.lstm))
    opt.lstm = test_opt.lstm
    print(f'Using lstm -> ', opt.lstm)

    eval_pixset(opt, epoch)
