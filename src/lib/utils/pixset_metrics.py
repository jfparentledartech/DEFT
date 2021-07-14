import numpy as np
import json

from matplotlib import pyplot as plt
from pioneer.common.IoU3d import matrixIoU

DEBUG = False

map_categories_to_id = {
    'pedestrian': 0,
    'bicycle': 1,
    'car': 2,
    'van': 3,
    'bus': 4,
    'truck': 5,
    'motorcycle': 6,
    'stop sign': 7,
    'traffic light': 8,
    'traffic sign': 9,
    'traffic cone': 10,
    'fire hydrant': 11,
    'cyclist': 12,
    'motorcyclist': 13,
    'unclassified vehicle': 14,
    'trailer': 15,
    'construction vehicle': 16,
    'barrier': 17
}

map_id_to_category = inv_map = {v: k for k, v in map_categories_to_id.items()}

def category_max_range(config):
    return [config['class_range']['pedestrian'], config['class_range']['car']]


def compute_metrics(ground_truths, hypothesis, eval_type='distance', im=None, category=None):
    config = load_evaluation_configuration()

    gt_to_keep = []
    for gt in ground_truths:
        max_range = category_max_range(config)[gt['category_id']-1]
        if gt['depth'].item() <= max_range:
            if category is not None:
                if map_id_to_category[gt['category_id'].item()-1] == category:
                    gt_to_keep.append(gt)
            else:
                gt_to_keep.append(gt)
    ground_truths = gt_to_keep

    hyp_to_keep = []
    for hyp in hypothesis:
        max_range = category_max_range(config)[map_categories_to_id[hyp.classe]]
        if hyp.depth.item() <= max_range:
            if category is not None:
                if hyp.classe == category:
                    hyp_to_keep.append(hyp)
            else:
                hyp_to_keep.append(hyp)
    hypothesis = hyp_to_keep

    gt_list = list(range(1,len(ground_truths)+1))
    hyp_list = list(range(1,len(hypothesis)+1))

    if eval_type == 'distance':
        distances = []
        for gt in range(len(gt_list)):
            gt_centroid = np.asarray(ground_truths[gt]['location'])
            # gt_centroid = np.asarray(ground_truths[gt]['amodel_center'])

            distances.append([])
            hyp_centroids = []
            for hyp in range(len(hyp_list)):
                hyp_centroid = hypothesis[hyp].ddd_bbox[3:-1]
                # hyp_centroid = hypothesis[hyp].ct
                x, y, w, h = hypothesis[hyp].tlwh
                hyp_centroid_ct = (x + w / 2, (y + h / 2))
                hyp_centroids.append(hyp_centroid_ct)
                distances[gt].append(distance(gt_centroid, hyp_centroid, config['dist_th_tp']))

        if DEBUG:
            gt_x, gt_y = np.asarray([ground_truths[i]['amodel_center'][0] for i in range(len(ground_truths))]), np.asarray([ground_truths[i]['amodel_center'][1] for i in range(len(ground_truths))])
            # gt_x, gt_y = np.asarray([ground_truths[i]['location'][0] for i in range(len(ground_truths))]), np.asarray([ground_truths[i]['location'][1] for i in range(len(ground_truths))])
            # hyp_x, hyp_y = np.asarray([hypothesis[i].ct[0] for i in range(len(hypothesis))]), np.asarray([hypothesis[i].ct[1] for i in range(len(hypothesis))])
            hyp_x, hyp_y = np.asarray(hyp_centroids)[:,0], np.asarray(hyp_centroids)[:,1]
            # hyp_x, hyp_y = np.asarray([hypothesis[i].ddd_bbox[3:-1][0] for i in range(len(hypothesis))]), np.asarray([hypothesis[i].ddd_bbox[3:-1][1] for i in range(len(hypothesis))])
            plt.scatter(gt_x, gt_y-389, c='r')
            plt.scatter(hyp_x, hyp_y, c='b')
            plt.imshow(im)
            plt.show()

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

        iou_threshold = 0.5
        iou_matrix[np.where(iou_matrix < iou_threshold)] = np.nan
        distances = 1-iou_matrix

    return gt_list, hyp_list, distances


def distance(point1, point2, max_distance):
    # d = np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2])) # TODO verify
    d = np.linalg.norm(np.array(point1[1:]) - np.array(point2[1:]))
    if d < max_distance:
        return d
    else:
        return np.nan


def load_evaluation_configuration():
    # with open('tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/configs/tracking_nips_2019.json') as f:
    with open('src/tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/configs/tracking_nips_2019.json') as f:
      return json.load(f)
