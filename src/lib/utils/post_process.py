# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Xingyi Zhou (zhouxy@cs.utexas.edu)
# Source: https://github.com/xingyizhou/CenterTrack/blob/master/src/lib/utils/post_process.py
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from .image import transform_preds_with_trans, get_affine_transform, transform_preds
from .ddd_utils import ddd2locrot, comput_corners_3d
from .ddd_utils import project_to_image, rot_y2alpha
import numba


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def generic_post_process(
    opt, dets, c, s, h, w, num_classes, calibs=None, height=-1, width=-1, distort_coeffs=None
):
    if not ("scores" in dets):
        return [{}], [{}]
    ret = []

    for i in range(len(dets["scores"])):
        preds = []
        trans = get_affine_transform(c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
        for j in range(len(dets["scores"][i])):
            if dets["scores"][i][j] < opt.out_thresh:
                break
            item = {}
            item["score"] = dets["scores"][i][j]
            item["class"] = int(dets["clses"][i][j]) + 1
            item["ct"] = transform_preds_with_trans(
                (dets["cts"][i][j]).reshape(1, 2), trans
            ).reshape(2)

            if "tracking" in dets:
                tracking = transform_preds_with_trans(
                    (dets["tracking"][i][j] + dets["cts"][i][j]).reshape(1, 2), trans
                ).reshape(2)
                item["tracking"] = tracking - item["ct"]

            if "bboxes" in dets:
                bbox = transform_preds_with_trans(
                    dets["bboxes"][i][j].reshape(2, 2), trans
                ).reshape(4)
                item["bbox"] = bbox

            if "hps" in dets:
                pts = transform_preds_with_trans(
                    dets["hps"][i][j].reshape(-1, 2), trans
                ).reshape(-1)
                item["hps"] = pts

            if "dep" in dets and len(dets["dep"][i]) > j:
                item["dep"] = dets["dep"][i][j]

            if "dim" in dets and len(dets["dim"][i]) > j:
                item["dim"] = dets["dim"][i][j]

            if "rot" in dets and len(dets["rot"][i]) > j:
                item["alpha"] = get_alpha(dets["rot"][i][j : j + 1])[0]

            if (
                "rot" in dets
                and "dep" in dets
                and "dim" in dets
                and len(dets["dep"][i]) > j
            ):
                if False and "amodel_offset" in dets and len(dets["amodel_offset"][i]) > j:
                    ct_output = dets["bboxes"][i][j].reshape(2, 2).mean(axis=0)
                    amodel_ct_output = ct_output + dets["amodel_offset"][i][j]
                    ct = (
                        transform_preds_with_trans(
                            amodel_ct_output.reshape(1, 2), trans
                        )
                        .reshape(2)
                        .tolist()
                    )
                else:
                    bbox = item["bbox"]
                    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                item["ct"] = ct

                ct[1] += 389

                item["loc"], item["rot_y"] = ddd2locrot(
                    ct, item["alpha"], item["dim"], item["dep"], calibs[i], distort_coeffs[i]
                )

            preds.append(item)

        if "nuscenes_att" in dets:
            for j in range(len(preds)):
                preds[j]["nuscenes_att"] = dets["nuscenes_att"][i][j]

        if "velocity" in dets:
            for j in range(len(preds)):
                preds[j]["velocity"] = dets["velocity"][i][j]

        ret.append(preds)

    return ret


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate(
                [
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()
        ret.append(top_preds)
    return ret
