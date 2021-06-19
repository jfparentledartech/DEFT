import PIL
import json
import numpy as np
import copy

from pioneer.das.api import platform
from matplotlib import pyplot as plt
from PIL import Image
from progress.bar import Bar
from nuscenes.utils.kitti import KittiDB

from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from matplotlib.patches import Polygon
from pioneer.common import linalg


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


def box3d_to_bbox(box, image_sample, annotation_sample):
    T = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
    vertices = linalg.bbox_to_8coordinates(box['c'], box['d'], box['r'])
    vertices_transformed = image_sample.transform_pts(T, vertices)

    p, mask_fov = image_sample.project_pts(vertices_transformed, mask_fov=False, output_mask=True, undistorted=False,
                                     margin=300)
    imcorners = [p[:, 0], p[:,1]]
    bbox = (np.min(imcorners[0]).item(), np.min(imcorners[1]).item(), np.max(imcorners[0]).item(), np.max(imcorners[1]).item())
    return bbox


def p3d_box(box, image_sample, image, annotation_sample, projected_center):
    T = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
    # annotation_centers_transformed = image_sample.transform_pts(T, annotation_centers)

    box['d'][0] += 0 # length
    box['d'][1] += 0 # width
    box['d'][2] += 0 # height

    box['r'][0] += 0 # roll
    box['r'][1] += 0 # pitch
    box['r'][2] += 0 # yaw

    box['c'][0] += 0 # x
    box['c'][1] += 0 # y
    box['c'][2] += 0 # z

    vertices = linalg.bbox_to_8coordinates(box['c'], box['d'], box['r'])
    vertices_transformed = image_sample.transform_pts(T, vertices)

    p, mask_fov = image_sample.project_pts(vertices_transformed, mask_fov=False, output_mask=True, undistorted=False,
                                     margin=300)
    imcorners = [p[:, 0], p[:,1]]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))
    xmin, ymin, xmax, ymax = bbox
    x_list = [xmin, xmin, xmax, xmax]
    y_list = [ymin, ymax, ymin, ymax]
    fig, ax = plt.subplots()
    ax.imshow(image)
    faces = [[0, 1, 3, 2], [0, 1, 5, 4], [0, 2, 6, 4], [7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]]
    for face in faces:
        poly = np.vstack([p[face[0]], p[face[1]], p[face[2]], p[face[3]], p[face[0]]])
        patch = Polygon(poly, closed=True, linewidth=1, edgecolor='b', facecolor='none')
        # Create figure and axes
        ax.add_patch(patch)
    plt.scatter(projected_center[0], projected_center[1])
    plt.scatter(x_list, y_list)
    plt.show()


if __name__ == '__main__':

    pixset_data_path = '/home/jfparent/Documents/Stage/DEFT/data/pixset/'
    pixset_images_path = pixset_data_path + 'v1.0-trainval/'
    pixset_annotations_path = pixset_data_path + 'annotations/'

    sync_labels = ['*ech*', '*_img*', '*_flimg*', '*_ftrr*', '*deepen*']
    interp_labels = ['*_xyzit*', 'sbgekinox_*', 'peakcan_*', '*temp', '*_xyzvcfar']
    tolerance_us = 2000

    dataset_path = '/home/jfparent/Documents/PixSet/20200721_180421_part41_1800_2500'

    pf = platform.Platform(dataset_path)
    sc = pf.synchronized(sync_labels=sync_labels, interp_labels=interp_labels, tolerance_us=tolerance_us)

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
        'motorcycle': filtered_categories[1]
    }

    categories_info = [{"name": list(filtered_categories)[i], "id": i+1} for i in range(len(filtered_categories))]

    coco_format = {
        'images': [],
        'annotations': [],
        'categories': categories_info,
        'videos': []
    }

    dataset = dataset_path.split("/")[-1]

    bar = Bar(f'Exporting {dataset} frames', max=len(sc))

    num_image = 0
    num_annotation = 0
    num_video = 0

    for i_frame, frame in enumerate(range(len(sc))):
        bar.next()
        image_sample = sc[frame][img_ds]
        image = Image.fromarray(image_sample.raw)
        image_path = f'{pixset_images_path}{dataset}_img{i_frame+1:06d}.jpg'
        image.save(image_path)

        annotation_sample = sc[frame][annotations]
        T = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
        annotation_centers = annotation_sample.raw['data']['c']
        annotation_centers_transformed = annotation_sample.transform_pts(T, annotation_centers)
        _, annotation_mask = image_sample.project_pts(annotation_centers_transformed, mask_fov=True, output_mask=True)
        projected_centers = image_sample.project_pts(annotation_centers_transformed, mask_fov=False)

        num_image += 1

        # TODO trans_matrix ->
        trans_matrix = [[-1.04343849e-01, -1.16873616e-02,  9.94472607e-01,  2.51601431e+02],
                        [-9.94372194e-01, -1.72119027e-02, -1.04535593e-01,  9.17405742e+02],
                        [ 1.83385110e-02, -9.99783555e-01, -9.82563118e-03,  1.50348339e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

        camera_intrinsic = np.column_stack((image_sample.camera_matrix, np.zeros(3))).tolist()

        coco_format['images'].append({
            "id": num_image,
            "file_name": image_path,
            "video_id": num_video,
            "frame_id": i_frame,
            "width": image_sample.raw.shape[1],
            "height": image_sample.raw.shape[0],
            "trans_matrix": trans_matrix,
            "calib": camera_intrinsic,
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

            center_coordinates = detection['c']
            # original_box_dimensions = detection['d']
            box_dimensions = [detection['d'][1], detection['d'][0], detection['d'][2]] # width, length, height
            yaw = detection['r'][2]
            category_id = filtered_categories.index(map_categories[categories[int(detection['classes'])]]) + 1
            object_id = detection['id']
            flags = detection['flags']

            attributes = annotation_sample.raw['attributes'][i_detection]

            # p3d_box(detection, image_sample, image, annotation_sample, projected_centers[i_detection])

            num_annotation += 1
            # instance information in COCO format
            coco_format['annotations'].append({
                "id": num_annotation,
                "image_id": num_image,
                "category_id": category_id,
                "dim": [box_dimensions[0].item(), box_dimensions[1].item(), box_dimensions[2].item()],
                "location": center_coordinates.tolist(), # TODO verify (model output negative values for x?)
                "depth": center_coordinates[0].item(),
                "occluded": int(attributes['occlusions']),
                "truncated": int(attributes['truncations']),
                "rotation_y": yaw.item(),
                "amodel_center": projected_centers[i_detection].tolist(),
                "iscrowd": 0,
                "track_id": int(object_id),
                # "attributes": ATTRIBUTE_TO_ID[att],
                # "velocity": vel
            })

            bbox = box3d_to_bbox(detection, image_sample, annotation_sample)
            alpha = _rot_y2alpha(
                yaw,
                (bbox[0] + bbox[2]) / 2,
                T[0, 2],
                T[0, 0],
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

    # TODO more than one video/scene
    num_video += 1
    coco_format["videos"].append({"id": num_video, "file_name": f'scene-{num_video:04d}'})

    out_path = pixset_annotations_path + "train.json"
    print("out_path", out_path)
    json.dump(coco_format, open(out_path, "w")) # TODO uncomment

    bar.finish()

