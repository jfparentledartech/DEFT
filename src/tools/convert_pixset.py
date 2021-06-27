import PIL
import json
import numpy as np
import copy
import cv2

from pioneer.das.api import platform
from matplotlib import pyplot as plt
from PIL import Image
from progress.bar import Bar
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.data_classes import Box

from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from matplotlib.patches import Polygon
from pioneer.common import linalg
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from lib.utils.ddd_utils import compute_box_3d, project_to_image, draw_box_3d, unproject_2d_to_3d, ddd2locrot

DEBUG = True
UNDISTORT = False

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


def project_pts(pts, camera_matrix):
    pts = pts.T
    R = T = np.zeros((3, 1))
    image_pts, _ = cv2.projectPoints(pts, R, T, np.asarray(camera_matrix)[:,:3], np.zeros((5, 1)))
    image_pts = np.squeeze(image_pts)
    return image_pts


def box3d_to_bbox(box, image_sample, annotation_sample):
    T = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
    vertices = linalg.bbox_to_8coordinates(box['c'], box['d'], box['r'])
    vertices_transformed = image_sample.transform_pts(T, vertices)
    p = image_sample.project_pts(vertices_transformed, mask_fov=False, undistorted=False)
    imcorners = [p[:, 0], p[:,1]]
    return (
        np.min(imcorners[0]).item(),
        np.min(imcorners[1]).item(),
        np.max(imcorners[0]).item(),
        np.max(imcorners[1]).item(),
    )


def p3d_box(box, image_sample, image, annotation_sample, amodel_center, projected_center, loc, dim, rot):
    annotation_to_camera_transformation = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
    # camera_to_annotation_transformation = image_sample.compute_transform(referential_or_ds=annotation_sample.label, ignore_orientation=True)
    camera_to_annotation_transformation = np.linalg.inv(annotation_to_camera_transformation)

    box['d'][0] += 0 # length
    box['d'][1] += 0 # width
    box['d'][2] += 0 # height

    box['r'][0] += 0 # roll
    box['r'][1] += 0 # pitch
    box['r'][2] += 0 # yaw

    box['c'][0] += 0 # x
    box['c'][1] += 0 # y
    box['c'][2] += 0 # z

    dim.reverse()
    loc = (camera_to_annotation_transformation @ np.asarray([loc[0], loc[1], loc[2], 1]))[:3]
    loc[2] += dim[2]/2

    # vertices = linalg.bbox_to_8coordinates(box['c'], box['d'], box['r'])
    vertices_label = linalg.bbox_to_8coordinates(box['c'], box['d'], [0,0,box['r'][2]])
    # vertices_pred = linalg.bbox_to_8coordinates(loc, dim, [0,0,box['r'][2]])
    R = -Rotation.from_matrix(camera_to_annotation_transformation[:3,:3] @ rot_y_axis(rot)).as_euler('xyz')[1]
    vertices_pred = linalg.bbox_to_8coordinates(loc, dim, [0,0,R])
    # vertices_pred = compute_box_3d(np.asarray(ann["dim"]), loc, box['r'][2])

    vertices_transformed_label = image_sample.transform_pts(annotation_to_camera_transformation, vertices_label)
    vertices_transformed_pred = image_sample.transform_pts(annotation_to_camera_transformation, vertices_pred)

    if UNDISTORT:
        p_label = image_sample.project_pts(vertices_transformed_label, undistorted=True)
        p_pred = image_sample.project_pts(vertices_transformed_pred, undistorted=True)
    else:
        p_label = image_sample.project_pts(vertices_transformed_label, undistorted=False)
        p_pred = image_sample.project_pts(vertices_transformed_pred, undistorted=False)
    # p_pred = project_to_image(box_3d, calib)


    imcorners = [p_label[:, 0], p_label[:,1]]
    # imcorners = [p[:, 0], p[:,1]]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))
    xmin, ymin, xmax, ymax = bbox
    x_list = [xmin, xmin, xmax, xmax]
    y_list = [ymin, ymax, ymin, ymax]
    fig, ax = plt.subplots()
    ax.imshow(image)
    faces = [[0, 1, 3, 2], [0, 1, 5, 4], [0, 2, 6, 4], [7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]]
    for face in faces:
        poly_label = np.vstack([p_label[face[0]], p_label[face[1]], p_label[face[2]], p_label[face[3]], p_label[face[0]]])
        poly_pred = np.vstack([p_pred[face[0]], p_pred[face[1]], p_pred[face[2]], p_pred[face[3]], p_pred[face[0]]])
        patch_label = Polygon(poly_label, closed=True, linewidth=1, edgecolor='b', facecolor='none')
        patch_pred = Polygon(poly_pred, closed=True, linewidth=1, edgecolor='r', facecolor='none')
        # Create figure and axes
        ax.add_patch(patch_label)
        ax.add_patch(patch_pred)
    plt.scatter(amodel_center[0], amodel_center[1], c='r')
    plt.scatter(projected_center[0], projected_center[1], c='b')
    plt.scatter(x_list, y_list)
    plt.show()

    p_pred[[1, 6]] = p_pred[[6, 1]]
    p_pred[[3, 4]] = p_pred[[4, 3]]
    p_pred[[1, 0]] = p_pred[[0, 1]]
    p_pred[[1, 3]] = p_pred[[3, 1]]
    p_pred[[4, 7]] = p_pred[[7, 4]]
    p_pred[[2, 3]] = p_pred[[3, 2]]

    return p_pred


def box3d_from_loc_dim_rot(image_sample, annotation_sample, loc, dim, rot):
    annotation_to_camera_transformation = annotation_sample.compute_transform(referential_or_ds=image_sample.label,
                                                                              ignore_orientation=True)
    camera_to_annotation_transformation = np.linalg.inv(annotation_to_camera_transformation)


    dim.reverse()
    loc = (camera_to_annotation_transformation @ np.asarray([loc[0], loc[1], loc[2], 1]))[:3]
    loc[2] += dim[2] / 2

    R = -Rotation.from_matrix(camera_to_annotation_transformation[:3, :3] @ rot_y_axis(rot)).as_euler('xyz')[1]
    vertices_pred = linalg.bbox_to_8coordinates(loc, dim, [0, 0, R])
    vertices_transformed_pred = image_sample.transform_pts(annotation_to_camera_transformation, vertices_pred)

    if UNDISTORT:
        p_pred = image_sample.project_pts(vertices_transformed_pred, undistorted=True)
    else:
        p_pred = image_sample.project_pts(vertices_transformed_pred, undistorted=False)


    p_pred[[1, 6]] = p_pred[[6, 1]]
    p_pred[[3, 4]] = p_pred[[4, 3]]
    p_pred[[1, 0]] = p_pred[[0, 1]]
    p_pred[[1, 3]] = p_pred[[3, 1]]
    p_pred[[4, 7]] = p_pred[[7, 4]]
    p_pred[[2, 3]] = p_pred[[3, 2]]

    return p_pred



if __name__ == '__main__':

    pixset_data_path = '/home/jfparent/Documents/Stage/DEFT/data/pixset/'
    pixset_images_path = pixset_data_path + 'v1.0-trainval/'
    pixset_annotations_path = pixset_data_path + 'annotations/'

    sync_labels = ['*ech*', '*_img*', '*_flimg*', '*_ftrr*', '*deepen*']
    interp_labels = ['*_xyzit*', 'sbgekinox_*', 'peakcan_*', '*temp', '*_xyzvcfar']
    tolerance_us = 2000

    img_ds = 'flir_bfc_img-cyl'
    annotations = 'pixell_bfc_box3d-deepen'

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
        'pedestrian',
        'vehicle'
    ]

    map_categories = {
        'pedestrian': filtered_categories[0],
        'deformed pedestrian': filtered_categories[0],
        'car': filtered_categories[1],
        'van': filtered_categories[1],
        'bus': filtered_categories[1],
        'truck': filtered_categories[1],
        'motorcycle': filtered_categories[1],
        'construction vehicle': filtered_categories[1],
        'unclassified vehicle': filtered_categories[1]

    }

    categories_info = [{"name": list(filtered_categories)[i], "id": i + 1} for i in range(len(filtered_categories))]

    coco_format = {
        'images': [],
        'annotations': [],
        'categories': categories_info,
        'videos': []
    }

    dataset_paths = [
        '/home/jfparent/Documents/PixSet/20200721_180421_part41_1800_2500',
        '/home/jfparent/Documents/PixSet/20200706_171559_part27_1170_1370',
        '/home/jfparent/Documents/PixSet/20200706_162218_part21_4368_7230',
        '/home/jfparent/Documents/PixSet/20200706_144800_part25_1224_2100'
    ]

    num_image = 0
    num_annotation = 0
    num_video = 0

    for dataset_path in dataset_paths:

        pf = platform.Platform(dataset_path)
        sc = pf.synchronized(sync_labels=sync_labels, interp_labels=interp_labels, tolerance_us=tolerance_us)

        dataset = dataset_path.split("/")[-1]

        bar = Bar(f'Exporting {dataset} frames', max=len(sc))

        for i_frame, frame in enumerate(range(len(sc))):
            print(f"{i_frame+1}/{len(sc)}")
            bar.next()

            for i_camera, camera in enumerate(['flir_bfl_img', 'flir_bfr_img', 'flir_bfc_img']):
            # for i_camera, camera in enumerate(['flir_bfr_img']):
                image_sample = sc[frame][camera]
                image = Image.fromarray(image_sample.raw)
                image_path = f'{pixset_images_path}{dataset}_{camera}{i_frame+1:06d}.jpg'
                image.save(image_path)

                annotation_sample = sc[frame][annotations]
                annotation_to_camera_transformation = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
                camera_to_annotation_transformation = image_sample.compute_transform(referential_or_ds=annotation_sample.label, ignore_orientation=True)
                annotation_centers = annotation_sample.raw['data']['c']
                annotation_centers_in_camera_reference = transform_pts(annotation_to_camera_transformation, annotation_centers)

                _, annotation_mask = image_sample.project_pts(annotation_centers_in_camera_reference, mask_fov=True, output_mask=True)
                # projected_centers = image_sample.project_pts(annotation_centers_in_camera_reference, undistorted=True)

                num_image = (i_frame+1) + (i_camera * len(sc))

                trans_matrix = annotation_to_camera_transformation.tolist()

                if UNDISTORT:
                    camera_intrinsic = np.column_stack((image_sample.camera_matrix, np.zeros(3))).tolist()
                    projected_centers = image_sample.project_pts(annotation_centers_in_camera_reference, undistorted=False)
                else:
                    camera_intrinsic = np.column_stack((image_sample.und_camera_matrix, np.zeros(3))).tolist()
                    projected_centers = image_sample.project_pts(annotation_centers_in_camera_reference, undistorted=True)

                coco_format['images'].append({
                    "id": num_image,
                    "file_name": image_path,
                    "video_id": num_video + i_camera + 1,
                    "frame_id": i_frame + 1,
                    "width": image_sample.raw.shape[1],
                    "height": image_sample.raw.shape[0],
                    "trans_matrix": trans_matrix,
                    "calib": camera_intrinsic
                    # "pose_record_trans": pose_record["translation"],
                    # "pose_record_rot": pose_record["rotation"],
                    # "cs_record_trans": cs_record["translation"],
                    # "cs_record_rot": cs_record["rotation"],
                })

                # plt.imshow(image)
                # plt.scatter(pts2d[:, 0], pts2d[:, 1])
                # plt.show()

                for i_detection, detection in enumerate(annotation_sample.raw['data']):
                    if not annotation_mask[i_detection]:
                        continue
                    if not categories[int(detection['classes'])] in list(map_categories.keys()):
                        continue

                    '''Fields are: 
                        'c' (float, float, float): center coordinates
                        'd' (float, float, float): the box dimensions (depth, width, height) TODO
                        'r' (float, float, float): the Euler angles (rotations)
                        'classes' (int): the object category number
                        'id' (int): the object's instance unique id 
                        'flags' (int): miscellaneous infos
                    Coordinate system: +x is forward, +y is left and +z is up. 
                    '''

                    # center_coordinates = detection['c']
                    # center_coordinates = np.asarray([detection['c'][2], detection['c'][1], detection['c'][0]])
                    center_coordinates = annotation_centers_in_camera_reference[i_detection]
                    # center_coordinates = annotation_centers[i_detection]

                    # original_box_dimensions = detection['d']
                    box_dimensions = detection['d'].tolist()
                    box_dimensions.reverse()
                    initial_yaw = detection['r'][2]

                    R_in_annotation_reference = rot_z_axis(detection['r'][2])
                    # R_in_annotation_reference = Rotation.from_euler('xyz', detection['r'], degrees=False).as_matrix()
                    R_in_camera_reference = np.dot(annotation_to_camera_transformation[:3,:3], R_in_annotation_reference)
                    v = np.dot(R_in_camera_reference, np.array([-1, 0, 0]))
                    yaw = -np.arctan2(v[2], v[0])
                    # yaw = detection['r'][2]
                    # center_coordinates[1] += box_dimensions[2] / 2
                    # box.translate(np.array([0, box.wlh[2] / 2, 0]))

                    category_id = filtered_categories.index(map_categories[categories[int(detection['classes'])]]) + 1
                    object_id = detection['id']
                    flags = detection['flags']

                    attributes = annotation_sample.raw['attributes'][i_detection]

                    # TODO difference?
                    if UNDISTORT:
                        amodel_center = image_sample.project_pts(center_coordinates, undistorted=True).tolist()
                    else:
                        # TODO verify
                        # when undistorted is true, dasapi unproject unproejction work.
                        # when undistorted is false, deft unproject workish.
                        # amodel_center = image_sample.project_pts(center_coordinates, undistorted=False).tolist()
                        amodel_center = image_sample.project_pts(center_coordinates, undistorted=True).tolist()
                    # amodel_center = project_to_image(
                    #     np.array(
                    #         [
                    #             center_coordinates[0],
                    #             center_coordinates[1],# - box_dimensions[2] / 2,
                    #             center_coordinates[2],
                    #         ],
                    #         np.float32,
                    #     ).reshape(1, 3),
                    #     np.asarray(camera_intrinsic),
                    # )[0].tolist()

                    num_annotation += 1
                    # instance information in COCO format
                    coco_format['annotations'].append({
                        "id": num_annotation,
                        "image_id": num_image,
                        "category_id": category_id,
                        "dim": box_dimensions,
                        "location": center_coordinates.tolist(), # TODO verify
                        "depth": center_coordinates[2].item(),
                        "occluded": int(attributes['occlusions']),
                        "truncated": int(attributes['truncations']),
                        "rotation_y": yaw.item(),
                        # "amodel_center": projected_centers[i_detection].tolist(),
                        "amodel_center": amodel_center,
                        "iscrowd": 0,
                        "track_id": int(object_id),
                        # "attributes": ATTRIBUTE_TO_ID[att],
                        # "velocity": vel
                    })

                    bbox = box3d_to_bbox(detection, image_sample, annotation_sample)
                    # box = Box(
                    #     center=center_coordinates,
                    #     size=np.asarray(box_dimensions),
                    #     orientation= Quaternion(Rotation.from_matrix(R_in_camera_reference).as_quat())
                    # )
                    # bbox = KittiDB.project_kitti_box_to_image(
                    #     copy.deepcopy(box), np.asarray(camera_intrinsic), imsize=(1440, 1080)
                    # )
                    alpha = _rot_y2alpha(
                        yaw,
                        (bbox[0] + bbox[2]) / 2,
                        np.asarray(camera_intrinsic)[0, 2],
                        np.asarray(camera_intrinsic)[0, 0],
                    )

                    coco_format['annotations'][-1]["bbox"] = [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ]
                    coco_format['annotations'][-1]["area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    coco_format['annotations'][-1]["alpha"] = alpha.item()

                    num_annotation += 1
                    if DEBUG:
                    # if i_frame==200:
                    # if i_frame%20 == 0:
                        calib = np.asarray(camera_intrinsic)
                        ann = coco_format['annotations'][-1]

                        # dist_coeff = np.asarray(image_sample.distortion_coeffs)
                        loc, rot = ddd2locrot(ann["amodel_center"], ann["alpha"], ann["dim"], ann["depth"], calib)
                        print('location', ann["location"])
                        print('unproject location', loc)
                        print('yaw', ann["rotation_y"])
                        print('unproject alpha2rot_y', rot)

                        box_3d = compute_box_3d(np.asarray(ann["dim"]), loc, rot)
                        box_2d = project_to_image(box_3d, calib)
                        im = np.ascontiguousarray(np.copy(image))
                        # # im = draw_box_3d(im, box_2d, same_color=True)
                        # box_3d = p3d_box(detection, image_sample, image, annotation_sample, amodel_center, projected_centers[i_detection],
                        #         loc, ann["dim"], rot)
                        box3d2 = box3d_from_loc_dim_rot(image_sample, annotation_sample, loc, ann["dim"], rot)

                        im = draw_box_3d(im, box3d2, same_color=True)
                        plt.imshow(im)
                        plt.show()

                        print()

        for i_camera, camera in enumerate(['flir_bfl_img', 'flir_bfr_img', 'flir_bfc_img']):
            coco_format["videos"].append({"id": num_video + i_camera+1, "file_name": f'scene-{num_video + i_camera+1:04d}'})
        num_video += 1
        bar.finish()

    out_path = pixset_annotations_path + "train.json"
    print("out_path", out_path)
    json.dump(coco_format, open(out_path, "w"))

