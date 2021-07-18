import numpy as np
import json

from matplotlib import pyplot as plt
from pioneer.common.IoU3d import matrixIoU

DEBUG = False

map_categories_to_id = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'trailer': 3,
    'pedestrian': 4,
    'motorcyclist': 5,
    'cyclist': 6,
    'van': 7
}

map_id_to_category = inv_map = {v: k for k, v in map_categories_to_id.items()}


def category_max_range(config, category):
    if category in list(config['class_range'].keys()):
        return config['class_range'][category]
    else:
        return 40


def compute_metrics(ground_truths, hypothesis, img_info, eval_type='distance', im=None, category=None):
    config = load_evaluation_configuration()

    gt_to_keep = []
    for gt in ground_truths:
        gt_category = list(map_categories_to_id.keys())[gt['category_id'] - 1]
        if gt_category == category:
            max_range = category_max_range(config, gt_category)
            if gt['depth'].item() <= max_range:
                    gt_to_keep.append(gt)
    ground_truths = gt_to_keep

    hyp_to_keep = []
    for hyp in hypothesis:

        if outside_fov(img_info, hyp.tlwh, im):
            continue

        if hyp.classe == category:
            max_range = category_max_range(config, hyp.classe)
            if hyp.depth.item() <= max_range:
                if category is not None:
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
    with open('tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/configs/tracking_nips_2019.json') as f:
    # with open('/home/jfparent/Documents/Stage/DEFT/src/tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/configs/tracking_nips_2019.json') as f:
        return json.load(f)

def outside_fov(img_info, tlwh, im):
    x, y, w, h = tlwh
    hyp_centroid_ct = (x + w / 2, (y + h / 2))
    hyp_centroid_left = (x, (y + h / 2))
    hyp_centroid_right = (x + w, (y + h / 2))

    crop_left = 0
    crop_right = 0
    cameras = ['flir_bfl_img', 'flir_bfr_img', 'flir_bfc_img']
    # cameras = ['flir_bfc_img']
    cam = cameras[img_info['sensor_id'].item()]
    if cam == 'flir_bfl_img':
        crop_left = 281
    elif cam == 'flir_bfr_img':
        crop_right = 281
    if DEBUG:
        print(x < crop_left or (x + w) > 1440 - crop_right)
        if (x < crop_left or (x + w) > 1440 - crop_right):
            plt.scatter(hyp_centroid_ct[0], hyp_centroid_ct[1])
            plt.scatter(hyp_centroid_left[0], hyp_centroid_left[1])
            plt.scatter(hyp_centroid_right[0], hyp_centroid_right[1])
            plt.imshow(im)
            plt.show()
    return x < crop_left or (x + w) > 1440 - crop_right
