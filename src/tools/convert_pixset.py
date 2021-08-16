import json
import numpy as np
import copy
import cv2
import os
import pickle

from pioneer.das.api import platform
from matplotlib import pyplot as plt
from PIL import Image
from progress.bar import Bar

from matplotlib.patches import Polygon
from pioneer.common import linalg
from pioneer.common.trace_processing import TraceProcessingCollection, Realign, ZeroBaseline, Smooth

import _init_paths
from lib.utils.ddd_utils import draw_box_3d, ddd2locrot

from uuid import uuid4

DEBUG = False

def rot_x_axis(theta):
    return np.asarray([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])

def rot_y_axis(theta):
    return np.asarray([[np.cos(theta), 0,np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])

def rot_z_axis(theta):
    return np.asarray([[np.cos(theta),-np.sin(theta),0],
                       [np.sin(theta),np.cos(theta),0],
                       [0,0,1]])

def transform_pts(matrix4x4, ptsNx3):
    return linalg.map_points(matrix4x4, ptsNx3)

def is_ndarray(object):
    return isinstance(object, np.ndarray)


def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def erroneous_projection(ann, calib, dist_coeffs):
    am_center = copy.copy(ann["amodel_center"])
    am_center[0] += crop_left
    am_center[1] += crop_top
    loc, rot = ddd2locrot(am_center, ann["alpha"], ann["dim"], ann["depth"], calib, dist_coeffs)
    projected_box3d = box3d_from_loc_dim_rot(annotation_to_camera_transformation, loc, ann["dim"], rot, calib,
                                             dist_coeffs)
    return np.min(projected_box3d[:,0] - crop_left) < 0 or np.max(projected_box3d[:,0] - crop_left) > 1440



def project_pts_dist_coeffs(pts, camera_matrix, dist_coeffs):
    pts = pts.T
    R = T = np.zeros((3, 1))
    image_pts, _ = cv2.projectPoints(pts, R, T, np.asarray(camera_matrix)[:,:3], dist_coeffs)
    image_pts = np.squeeze(image_pts)
    return image_pts


def box3d_to_bbox(detection, image_sample, annotation_to_camera_transformation):
    # T = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
    T = annotation_to_camera_transformation
    box3d_in_annotation_reference = linalg.bbox_to_8coordinates(detection['c'], detection['d'], detection['r']) # 3dbox coords in annotation reference
    box3d_in_camera_reference = transform_pts(T, box3d_in_annotation_reference)
    box2d = image_sample.project_pts(box3d_in_camera_reference)

    # box2d[:, 0] -= crop_left
    box2d[:, 1] -= crop_top

    imcorners = [box2d[:, 0], box2d[:,1]]
    return (
        np.min(imcorners[0]).item(),
        np.min(imcorners[1]).item(),
        np.max(imcorners[0]).item(),
        np.max(imcorners[1]).item(),
    )


def p3d_box(box, image_sample, image, annotation_sample, amodel_center, projected_center, loc, dim, rot):
    annotation_to_camera_transformation = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
    camera_to_annotation_transformation = np.linalg.inv(annotation_to_camera_transformation)

    dim_copy = copy.deepcopy(dim)
    dim_copy.reverse()

    loc_copy = (camera_to_annotation_transformation @ np.asarray([loc[0], loc[1], loc[2], 1]))[:3]
    loc_copy[2] += dim_copy[2]/2

    vertices_label = linalg.bbox_to_8coordinates(box['c'], box['d'], [0,0,box['r'][2]])
    R_in_camera_reference = annotation_to_camera_transformation[:3, :3] @ rot_z_axis(rot)
    v = np.dot(R_in_camera_reference, np.array([1, 0, 0]))
    yaw = -np.arctan2(v[2], v[0])
    print(yaw, box['r'][2])

    vertices_pred = linalg.bbox_to_8coordinates(loc_copy, dim_copy, [0,0,yaw])

    vertices_transformed_label = image_sample.transform_pts(annotation_to_camera_transformation, vertices_label)
    vertices_transformed_pred = image_sample.transform_pts(annotation_to_camera_transformation, vertices_pred)

    p_label = image_sample.project_pts(vertices_transformed_label, undistorted=False)
    p_pred = image_sample.project_pts(vertices_transformed_pred, undistorted=False)

    imcorners = [p_label[:, 0], p_label[:,1]]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))
    xmin, ymin, xmax, ymax = bbox
    x_list = [xmin, xmin, xmax, xmax]
    y_list = [ymin, ymax, ymin, ymax]

    fig, ax = plt.subplots()

    im = copy.copy(image)
    top_pad = np.zeros((crop_top, 1440, 3), dtype=int)
    bottom_pad = np.zeros((crop_bottom, 1440, 3), dtype=int)
    im = np.concatenate((top_pad, im, bottom_pad))
    ax.imshow(im)

    faces = [[0, 1, 3, 2], [0, 1, 5, 4], [0, 2, 6, 4], [7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]]
    for face in faces:
        poly_label = np.vstack([p_label[face[0]], p_label[face[1]], p_label[face[2]], p_label[face[3]], p_label[face[0]]])
        poly_pred = np.vstack([p_pred[face[0]], p_pred[face[1]], p_pred[face[2]], p_pred[face[3]], p_pred[face[0]]])
        patch_label = Polygon(poly_label, closed=True, linewidth=1, edgecolor='b', facecolor='none')
        patch_pred = Polygon(poly_pred, closed=True, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(patch_label)
        ax.add_patch(patch_pred)
    plt.scatter(amodel_center[0], amodel_center[1], c='r')
    plt.scatter(projected_center[0], projected_center[1], c='b')
    plt.scatter(x_list, y_list)
    # plt.show()

    p_pred[[1, 6]] = p_pred[[6, 1]]
    p_pred[[3, 4]] = p_pred[[4, 3]]
    p_pred[[1, 0]] = p_pred[[0, 1]]
    p_pred[[1, 3]] = p_pred[[3, 1]]
    p_pred[[4, 7]] = p_pred[[7, 4]]
    p_pred[[2, 3]] = p_pred[[3, 2]]

    return p_pred


def box3d_from_loc_dim_rot(annotation_to_camera_transformation, loc, dim, rot, camera_matrix, dist_coeffs):
    camera_to_annotation_transformation = np.linalg.inv(annotation_to_camera_transformation)
    dim_copy = copy.deepcopy(dim)
    dim_copy.reverse()
    loc = (camera_to_annotation_transformation @ np.asarray([loc[0], loc[1], loc[2], 1]))[:3]
    loc[2] += dim_copy[2] / 2

    R = annotation_to_camera_transformation[:3, :3] @ rot_z_axis(rot)
    v = np.dot(R, np.array([1, 0, 0]))
    yaw = -np.arctan2(v[2], v[0])
    box3d_pred_in_annotation_reference = linalg.bbox_to_8coordinates(loc, dim_copy, [0, 0, yaw])
    box3d_pred_in_camera_reference = transform_pts(annotation_to_camera_transformation, box3d_pred_in_annotation_reference)

    box3d = project_pts_dist_coeffs(box3d_pred_in_camera_reference, camera_matrix, dist_coeffs)

    # TODO refactor
    box3d[[1, 6]] = box3d[[6, 1]]
    box3d[[3, 4]] = box3d[[4, 3]]
    box3d[[1, 0]] = box3d[[0, 1]]
    box3d[[1, 3]] = box3d[[3, 1]]
    box3d[[4, 7]] = box3d[[7, 4]]
    box3d[[2, 3]] = box3d[[3, 2]]

    return box3d


if __name__ == '__main__':

    pixset_data_path = '/home/jfparent/Documents/Stage/DEFT/data/pixset/'
    pixset_images_path = pixset_data_path + 'v1.0-trainval/'
    pixset_pixell_path = pixset_data_path + 'pixell/'
    pixset_annotations_path = pixset_data_path + 'annotations/'

    sync_labels = ['*ech*', '*_img*', '*_flimg*', '*_ftrr*', '*deepen*']
    interp_labels = ['*_xyzit*', 'sbgekinox_*', 'peakcan_*', '*temp', '*_xyzvcfar']
    tolerance_us = 2000

    img_ds = 'flir_bfc_img-cyl'
    annotations = 'pixell_bfc_box3d-deepen'
    pixell = 'pixell_bfc_ftrr'

    categories = [
        'pedestrian',
        'deformed pedestrian',
        'bicycle',
        'car',
        'van',
        'bus',
        'truck',
        'motorcycle',
        'stop sign',
        'traffic light',
        'traffic sign',
        'traffic cone',
        'fire hydrant',
        'guard rail',
        'pole',
        'pole group',
        'road',
        'sidewalk',
        'wall',
        'building',
        'vegetation',
        'terrain',
        'ground',
        'crosstalk',
        'noise',
        'others',
        'animal',
        'unpainted',
        'cyclist',
        'motorcyclist',
        'unclassified vehicle',
        'obstacle',
        'trailer',
        'barrier',
        'bicycle rack',
        'construction vehicle'
    ]

    filtered_categories = [
        'car',
        'truck',
        'bus',
        'pedestrian',
        'motorcyclist',
        'cyclist',
        'van'
    ]

    map_categories = {
        'pedestrian': filtered_categories[3],
        'deformed pedestrian': filtered_categories[3],
        'cyclist': filtered_categories[5],
        'car': filtered_categories[0],
        'van': filtered_categories[6],
        'bus': filtered_categories[2],
        'truck': filtered_categories[1],
        'motorcyclist': filtered_categories[4]
    }

    categories_info = [{"name": list(filtered_categories)[i], "id": i + 1} for i in range(len(filtered_categories))]

    coco_format = {}
    coco_format['train'] = {
        'images': [],
        'annotations': [],
        'categories': categories_info,
        'videos': []
    }

    coco_format['val'] = {
        'images': [],
        'annotations': [],
        'categories': categories_info,
        'videos': []
    }

    coco_format['test'] = {
        'images': [],
        'annotations': [],
        'categories': categories_info,
        'videos': []
    }

    train_pixset_path = '/home/jfparent/Documents/PixSet/train_dataset/'
    # train_dataset_paths = [train_pixset_path+d for d in os.listdir(train_pixset_path)]

    test_pixset_path = '/home/jfparent/Documents/PixSet/test_dataset/'
    # test_dataset_paths = [test_pixset_path+d for d in os.listdir(test_pixset_path)]

    train_dataset_paths = [
        # '/home/jfparent/Documents/PixSet/train_dataset/20200721_180421_part41_1800_2500',
        '/home/jfparent/Documents/PixSet/train_dataset/20200706_171559_part27_1170_1370',
        # '/home/jfparent/Documents/PixSet/train_dataset/20200803_174859_part46_2761_2861', # rain
        # '/home/jfparent/Documents/PixSet/train_dataset/20200805_002607_part48_2083_2282', # night
        # '/home/jfparent/Documents/PixSet/train_dataset/20200721_164103_part43_3412_4100',
        # '/home/jfparent/Documents/PixSet/train_dataset/20200706_162218_part21_4368_7230',
        # '/home/jfparent/Documents/PixSet/train_dataset/20200706_144800_part25_1224_2100'
    ]

    test_dataset_paths = [
        # '/home/jfparent/Documents/PixSet/20200721_180421_part41_1800_2500',
        # '/home/jfparent/Documents/PixSet/train_dataset/20200706_171559_part27_1170_1370',
        # '/home/jfparent/Documents/PixSet/train_dataset/20200706_162218_part21_4368_7230',
        # '/home/jfparent/Documents/PixSet/20200706_144800_part25_1224_2100'
    ]

    category_counters = {
        'train': {},
        'val': {},
        'test': {}
    }

    num_image = 0
    num_annotation = 1

    for num_video, dataset_path in enumerate(train_dataset_paths+test_dataset_paths):

        pf = platform.Platform(dataset_path)
        sc = pf.synchronized(sync_labels=sync_labels, interp_labels=interp_labels, tolerance_us=tolerance_us)

        dataset = dataset_path.split("/")[-1]

        bar = Bar(f'Exporting {dataset} ({num_video+1}/{len(train_dataset_paths+test_dataset_paths)})', max=len(sc)*3)

        # for sensor_id, camera in enumerate(['flir_bfl_img', 'flir_bfr_img', 'flir_bfc_img']):
        for sensor_id, camera in enumerate(['flir_bfc_img']):

            for i_frame, frame in enumerate(range(len(sc))):
                if dataset_path in test_dataset_paths:
                    split = 'test'
                else:
                    if i_frame+1 > int(len(sc) * 0.2):
                        split = 'train'
                    else:
                        split = 'val'

                # print(f"{i_frame+1}/{len(sc)}")
                bar.next()

                waveform_sample = sc[frame][pixell]
                preprocess_list = [Realign(-10), ZeroBaseline()]
                waveform_processing = TraceProcessingCollection(preprocess_list)
                processed_waveform = waveform_sample.processed_array(waveform_processing)

                high_intensity_waveform = processed_waveform[0]
                low_intensity_waveform = processed_waveform[1][:,:,:256]
                full_waveform = np.concatenate((low_intensity_waveform, high_intensity_waveform), axis=2)
                # full_waveform = processed_waveform

                full_waveforms = {
                    'flir_bfl_img': np.concatenate((np.zeros((8,6,768)), full_waveform[:,:30,:]),axis=1),
                    'flir_bfc_img': full_waveform[:,30:66,:],
                    'flir_bfr_img': np.concatenate((full_waveform[:,:30,:], np.zeros((8,6,768))),axis=1)
                }

                camera_waveform_crops = (389,378,0,0)

                crop_top, crop_bottom, crop_left, crop_right = camera_waveform_crops
                side_crop = 281
                image_sample = sc[frame][camera]
                image_array = image_sample.raw

                if camera == 'flir_bfl_img':
                    image_array[:,:side_crop] = 0
                    crop_left = side_crop
                if camera == 'flir_bfr_img':
                    image_array[:,1440-side_crop:] = 0
                    crop_right = side_crop

                image = Image.fromarray(image_sample.raw[crop_top:-crop_bottom])

                num_image +=1
                image_path = f'{pixset_images_path}{dataset}_{camera}_{num_image:06d}.jpg'

                image.save(image_path)

                full_waveform = full_waveforms[camera]
                waveform_path = f'{pixset_pixell_path}{dataset}_{camera}_pixell_ftrr_{num_image:06d}.npy'
                np.save(waveform_path, full_waveform)

                annotation_sample = sc[frame][annotations]
                annotation_to_camera_transformation = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
                camera_to_annotation_transformation = image_sample.compute_transform(referential_or_ds=annotation_sample.label, ignore_orientation=True)
                annotation_centers = annotation_sample.raw['data']['c']
                annotation_centers_in_camera_reference = transform_pts(annotation_to_camera_transformation, annotation_centers)

                _, annotation_mask = image_sample.project_pts(annotation_centers_in_camera_reference, mask_fov=True, output_mask=True)

                trans_matrix = annotation_to_camera_transformation.tolist()

                camera_intrinsic = np.column_stack((image_sample.camera_matrix, np.zeros(3))).tolist()
                projected_centers = image_sample.project_pts(annotation_centers_in_camera_reference)

                coco_format[split]['images'].append({
                    "id": num_image,
                    # "file_name": image_path.replace('Documents/Stage/', ''),
                    "file_name": image_path,
                    "waveform_file_name": waveform_path,
                    "video_id": num_video,
                    "frame_id": i_frame + 1,
                    "width": image_sample.raw.shape[1],
                    "height": image_sample.raw.shape[0],
                    "sensor_id": sensor_id,
                    "sample_token": str(uuid4()).replace('-',''),
                    "trans_matrix": trans_matrix,
                    "calib": camera_intrinsic,
                    "camera_matrix": camera_intrinsic,
                    "distortion_coefficients": image_sample.distortion_coeffs.tolist()
                })

                for i_detection, detection in enumerate(annotation_sample.raw['data']):
                    if not annotation_mask[i_detection]:
                        continue
                    if not categories[int(detection['classes'])] in list(map_categories.keys()):
                        continue

                    attributes = annotation_sample.raw['attributes'][i_detection]
                    if attributes['occlusions'] == 2:
                        continue

                    amodel_center = projected_centers[i_detection].tolist()
                    if not (((amodel_center[0] - crop_left) < 1440) and ((amodel_center[0] - crop_left) > 0) and
                            ((amodel_center[1] - crop_top) < 313) and ((amodel_center[1] - crop_top) > 0)):
                        continue

                    amodel_center[1] -= crop_top

                    center_coordinates = annotation_centers_in_camera_reference[i_detection]
                    box_dimensions = detection['d'].tolist()
                    box_dimensions.reverse()

                    R_in_annotation_reference = rot_z_axis(detection['r'][2])
                    R_in_camera_reference = np.dot(annotation_to_camera_transformation[:3,:3], R_in_annotation_reference)
                    v = np.dot(R_in_camera_reference, np.array([-1, 0, 0]))
                    yaw = -np.arctan2(v[2], v[0])

                    category_id = filtered_categories.index(map_categories[categories[int(detection['classes'])]]) + 1
                    object_id = detection['id']

                    # instance information in COCO format
                    coco_format[split]['annotations'].append({
                        "id": num_annotation,
                        "image_id": num_image,
                        "category_id": category_id,
                        "dim": box_dimensions,
                        "location": center_coordinates.tolist(), # TODO verify
                        "depth": center_coordinates[2].item(),
                        "occluded": int(attributes['occlusions']),
                        "truncated": int(attributes['truncations']),
                        "rotation_y": yaw.item(),
                        "amodel_center": amodel_center,
                        "iscrowd": 0,
                        "track_id": int(object_id),
                    })

                    bbox = box3d_to_bbox(detection, image_sample, annotation_to_camera_transformation)
                    xmin, ymin, xmax, ymax = bbox
                    x_list = [xmin, xmin, xmax, xmax]
                    y_list = [ymin, ymax, ymin, ymax]

                    alpha = _rot_y2alpha(
                        yaw,
                        (bbox[0] + bbox[2]) / 2,
                        np.asarray(camera_intrinsic)[0, 2],
                        np.asarray(camera_intrinsic)[0, 0],
                        )

                    coco_format[split]['annotations'][-1]["bbox"] = [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        ]
                    coco_format[split]['annotations'][-1]["area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    coco_format[split]['annotations'][-1]["alpha"] = alpha.item()

                    dist_coeffs = np.asarray(coco_format[split]['images'][-1]['distortion_coefficients'])
                    ann = coco_format[split]['annotations'][-1]
                    calib = np.asarray(camera_intrinsic)

                    if erroneous_projection(ann, calib, dist_coeffs):
                        del coco_format[split]['annotations'][-1]
                    else:
                        num_annotation += 1

                        cat = categories[int(detection['classes'])]
                        if cat == 'deformed pedestrian':
                            cat = 'pedestrian'
                        if cat in category_counters[split].keys():
                            category_counters[split][cat] += 1
                        else:
                            category_counters[split][cat] = 1


                        if DEBUG:
                            print('Category: ' + categories[int(detection['classes'])])
                            am_center = copy.copy(ann["amodel_center"])
                            am_center[1] += crop_top
                            loc, rot = ddd2locrot(am_center, ann["alpha"], ann["dim"], ann["depth"], calib, dist_coeffs)
                            print('location', ann["location"])
                            print('unproject location', loc)
                            print('yaw', ann["rotation_y"])
                            print('unproject alpha2rot_y', rot)

                            box_3d = p3d_box(detection, image_sample, image, annotation_sample, am_center, projected_centers[i_detection],
                                             loc, ann["dim"], rot)

                            calib = coco_format[split]['images'][-1]['camera_matrix']
                            projected_box3d = box3d_from_loc_dim_rot(annotation_to_camera_transformation, loc, ann["dim"], rot, calib, dist_coeffs)

                            im = np.ascontiguousarray(np.copy(image))
                            top_pad = np.zeros((crop_top, 1440, 3), dtype=float)
                            bottom_pad = np.zeros((crop_bottom, 1440, 3), dtype=float)
                            try:
                                im = np.concatenate((top_pad, (im / 255), bottom_pad))
                            except:
                                print()
                            im = draw_box_3d(im, projected_box3d, same_color=True)
                            im = im[crop_top:-crop_bottom]
                            plt.imshow(im)
                            # plt.savefig(f'plt_{camera}_cropped_image.png')
                            # Image.fromarray(np.array(im * 255, dtype='uint8')).save(f'{camera}_cropped_image.png')
                            plt.show()
                            print()

        for sensor_id, camera in enumerate(['flir_bfl_img', 'flir_bfr_img', 'flir_bfc_img']):
            coco_format[split]["videos"].append({"id": num_video, "file_name": f'scene-{num_video:04d}'})
        bar.finish()

    train_path = pixset_annotations_path + "train.json"
    val_path = pixset_annotations_path + "val.json"
    test_path = pixset_annotations_path + "test.json"
    print('Saving train annotations...')
    json.dump(coco_format['train'], open(train_path, "w"))
    print('Saving val annotations...')
    json.dump(coco_format['val'], open(val_path, "w"))
    print('Saving test annotations...')
    json.dump(coco_format['test'], open(test_path, "w"))
    #
    # category_counters['train'] = dict(sorted(category_counters['train'].items(), key=lambda item: item[0]))
    # category_counters['val'] = dict(sorted(category_counters['val'].items(), key=lambda item: item[0]))
    # category_counters['test'] = dict(sorted(category_counters['test'].items(), key=lambda item: item[0]))
    #
    # N = len(list(category_counters['train'].keys()))
    # ind = np.arange(N)  # the x locations for the groups
    # width = 0.3  # the width of the bars
    #
    # _, ax = plt.subplots()
    # train_occurrence_ratios = np.asarray(list(category_counters['train'].values()))/np.sum(np.asarray(list(category_counters['train'].values())))
    # val_occurrence_ratios = np.asarray(list(category_counters['val'].values()))/np.sum(np.asarray(list(category_counters['val'].values())))
    # test_occurrence_ratios = np.asarray(list(category_counters['test'].values()))/np.sum(np.asarray(list(category_counters['test'].values())))
    # train_bars = ax.bar(ind,train_occurrence_ratios, width, color='r')
    # val_bars = ax.bar(ind + width, val_occurrence_ratios, width, color='b')
    # test_bars = ax.bar(ind + 2*width, test_occurrence_ratios, width, color='g')
    #
    # # add some text for labels, title and axes ticks
    # ax.set_ylabel('Occurrence ratios')
    # ax.set_title('Category occurrence ratios')
    # ax.set_xticks(ind + width / 3)
    # ax.set_xticklabels(category_counters['train'].keys(), rotation=20)
    #
    # ax.legend((train_bars[0], val_bars[0], test_bars[0]), ('Train', 'Val', 'Test'))
    # plt.savefig('category_occurrence_ratios.png')
    # plt.show()
    #
    # _, ax = plt.subplots()
    # val_occurrence_ratios = np.asarray(list(category_counters['val'].values()))/np.sum(np.asarray(list(category_counters['val'].values())))
    # test_occurrence_ratios = np.asarray(list(category_counters['test'].values()))/np.sum(np.asarray(list(category_counters['test'].values())))
    # train_bars = ax.bar(ind, category_counters['train'].values(), width, color='r')
    # val_bars = ax.bar(ind + width, category_counters['val'].values(), width, color='b')
    # test_bars = ax.bar(ind + 2*width, category_counters['test'].values(), width, color='g')
    #
    # # add some text for labels, title and axes ticks
    # ax.set_ylabel('Occurrences')
    # ax.set_title('Category occurrences')
    # ax.set_xticks(ind + width / 3)
    # ax.set_xticklabels(category_counters['train'].keys(), rotation=20)
    #
    # ax.legend((train_bars[0], val_bars[0], test_bars[0]), ('Train', 'Val', 'Test'))
    # plt.savefig('category_occurrence.png')
    # plt.show()


    print('Saving occurrences pickles...')
    with open('train_category_occurrences.pkl', 'wb') as f:
        pickle.dump(category_counters['train'], f)
    with open('val_category_occurrences.pkl', 'wb') as f:
        pickle.dump(category_counters['val'], f)
    with open('test_category_occurrences.pkl', 'wb') as f:
        pickle.dump(category_counters['test'], f)

    print('All done!')
