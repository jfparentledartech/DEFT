import numpy as np
import json

from matplotlib import pyplot as plt
from pioneer.common.IoU3d import matrixIoU

map_categories = {
    'pedestrian': 0,
    'vehicle': 1
}


def category_max_range(config):
    return [config['class_range']['pedestrian'], config['class_range']['car']]


def compute_metrics(ground_truths, hypothesis, eval_type='distance'):
    config = load_evaluation_configuration()

    gt_to_keep = []
    for gt in ground_truths:
        max_range = category_max_range(config)[gt['category_id']-1]
        if gt['depth'].item() <= max_range:
            gt_to_keep.append(gt)
    ground_truths = gt_to_keep

    hyp_to_keep = []
    for hyp in hypothesis:
        max_range = category_max_range(config)[map_categories[hyp.classe]]
        if hyp.depth.item() <= max_range:
            hyp_to_keep.append(hyp)
    hypothesis = hyp_to_keep

    gt_list = list(range(1,len(ground_truths)+1))
    hyp_list = list(range(1,len(hypothesis)+1))

    if eval_type == 'distance':
        distances = []
        for gt in range(len(gt_list)):
            gt_centroid = np.asarray(ground_truths[gt]['location'])
            distances.append([])
            for hyp in range(len(hyp_list)):
                hyp_centroid = hypothesis[hyp].ddd_bbox[3:-1]
                distances[gt].append(distance(gt_centroid, hyp_centroid, config['dist_th_tp']))

    if eval_type == 'iou':
        hypothesis_info = [
            np.asarray([np.asarray(hypothesis[i].ddd_bbox[3:-1]) for i in range(len(hypothesis))]),
            np.asarray([np.asarray(hypothesis[i].ddd_bbox[:3]) for i in range(len(hypothesis))]),'z',
            np.asarray([np.asarray(hypothesis[i].ddd_bbox[-1]) for i in range(len(hypothesis))]),
        ]

        ground_truth_info = [
            np.asarray([np.asarray(ground_truths[i]['location']) for i in range(len(ground_truths))]),
            np.asarray([np.asarray(ground_truths[i]['dim']) for i in range(len(ground_truths))]),'z',
            np.asarray([np.asarray(ground_truths[i]['rotation_y']) for i in range(len(ground_truths))]),
        ]

        iou_matrix = matrixIoU(
            ground_truth_info,
            hypothesis_info
        )

        iou_threshold = 0.25
        iou_matrix[np.where(iou_matrix < iou_threshold)] = np.nan
        distances = 1-iou_matrix

    return gt_list, hyp_list, distances


def distance(point1, point2, max_distance):
    d = np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))
    if d < max_distance:
        return d
    else:
        return np.nan


def load_evaluation_configuration():
    with open('tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/configs/tracking_nips_2019.json') as f:
      return json.load(f)
