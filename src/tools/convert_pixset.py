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

from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from matplotlib.patches import Polygon
from pioneer.common import linalg
from lib.utils.ddd_utils import compute_box_3d, project_to_image, draw_box_3d, unproject_2d_to_3d


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

    p, mask_fov = image_sample.project_pts(vertices_transformed, mask_fov=False, output_mask=True, undistorted=False,
                                     margin=300)
    imcorners = [p[:, 0], p[:,1]]
    bbox = (np.min(imcorners[0]).item(), np.min(imcorners[1]).item(), np.max(imcorners[0]).item(), np.max(imcorners[1]).item())
    return bbox


def p3d_box(box, image_sample, image, annotation_sample, projected_center, amodel_center):
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
    # [[7.90886694,  4.9315693, - 0.54329288],
    #  [7.79301908,  4.97551004,  1.07515882],
    # [7.9213846, 2.76998744, - 0.48371021],
    # [7.80553675,  2.81392819,  1.13474148],
    # [2.939363, 4.89301485, - 0.89796039],
    # [2.82351515,  4.93695559,  0.7204913],
    # [2.95188067,  2.73143299, - 0.83837773],
    # [2.83603281,  2.77537374, 0.78007397]]
    #
    # [[4.5415545,  3.8534715, - 2.3663845],
    #  [4.580207,   3.8534715,  2.6157584],
    # [6.2033453,  3.8534715,  2.6031656]
    # [6.164693,   3.8534715, - 2.3789773],
    # [4.5415545,  1.6910324, - 2.3663845],
    # [4.580207,  1.6910324,  2.6157584],
    # [6.2033453, 1.6910324,  2.6031656],
    # [6.164693,   1.6910324, - 2.3789773]]

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
    plt.scatter(amodel_center[0], amodel_center[1])
    plt.scatter(x_list, y_list)
    plt.show()


# def test_3d_params(dim, loc, rot, calib, image):
#     # dim = (height, width, length)
#     box_3d = compute_box_3d(dim, loc, rot)
#
#     # box_3d = np.asarray([[7.90886694,  4.9315693, - 0.54329288],
#     #                      [7.79301908,  4.97551004,  1.07515882],
#     #                      [7.9213846, 2.76998744, - 0.48371021],
#     #                      [7.80553675,  2.81392819,  1.13474148],
#     #                      [2.939363, 4.89301485, - 0.89796039],
#     #                      [2.82351515,  4.93695559 , 0.7204913],
#     #                      [2.95188067,  2.73143299, - 0.83837773],
#     #                      [2.83603281,  2.77537374, 0.78007397]])
#
#
#     box_2d = project_to_image(box_3d, calib)
#     # box_2d = project_to_image(box_3d, (np.asarray(calib)*10).tolist())
#     im = draw_box_3d(image, box_2d, c='b', same_color=True)
#     plt.imshow(im)
#     plt.show()
#     print()


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
        # image_sample = sc[frame][img_ds]
        image_sample = sc[frame]['flir_bfc_img']
        image = Image.fromarray(image_sample.raw)
        image_path = f'{pixset_images_path}{dataset}_img{i_frame+1:06d}.jpg'
        image.save(image_path)

        annotation_sample = sc[frame][annotations]
        annotation_to_camera_transformation = annotation_sample.compute_transform(referential_or_ds=image_sample.label, ignore_orientation=True)
        # camera_to_annotation_transformation = np.linalg.inv(annotation_to_camera_transformation)
        annotation_centers = annotation_sample.raw['data']['c']
        annotation_centers_in_camera_reference = transform_pts(annotation_to_camera_transformation, annotation_centers)

        _, annotation_mask = image_sample.project_pts(annotation_centers_in_camera_reference, mask_fov=True, output_mask=True)
        projected_centers = image_sample.project_pts(annotation_centers_in_camera_reference)

        num_image += 1

        trans_matrix = annotation_to_camera_transformation.tolist()
        camera_intrinsic = np.column_stack((image_sample.camera_matrix, np.zeros(3))).tolist()

        coco_format['images'].append({
            "id": num_image,
            "file_name": image_path,
            "video_id": num_video,
            "frame_id": i_frame,
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

            # original_box_dimensions = detection['d']
            # box_dimensions = [detection['d'][1], detection['d'][0], detection['d'][2]] # width, length, height
            box_dimensions = [detection['d'][2], detection['d'][1], detection['d'][0]] # width, length, height
            yaw = detection['r'][2]
            category_id = filtered_categories.index(map_categories[categories[int(detection['classes'])]]) + 1
            object_id = detection['id']
            flags = detection['flags']

            attributes = annotation_sample.raw['attributes'][i_detection]

            dim = box_dimensions
            loc = center_coordinates.tolist()
            rot = yaw.item()
            calib = camera_intrinsic
            # test_3d_params(dim, loc, rot, calib, image)

            amodel_center = project_pts(np.asarray([center_coordinates]), calib).tolist()
            # amodel_center = project_to_image(
            #     np.array(
            #         [
            #             center_coordinates[0],
            #             center_coordinates[1],# - box_dimensions[2] / 2,
            #             center_coordinates[2],
            #         ],
            #         np.float32,
            #     ).reshape(1, 3),
            #     calib,
            # )[0].tolist()

            num_annotation += 1
            # instance information in COCO format
            coco_format['annotations'].append({
                "id": num_annotation,
                "image_id": num_image,
                "category_id": category_id,
                "dim": [box_dimensions[0].item(), box_dimensions[1].item(), box_dimensions[2].item()],
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
            # p3d_box(detection, image_sample, image, annotation_sample, projected_centers[i_detection], amodel_center)

            bbox = box3d_to_bbox(detection, image_sample, annotation_sample)
            alpha = _rot_y2alpha(
                yaw,
                (bbox[0] + bbox[2]) / 2,
                annotation_to_camera_transformation[0, 2],
                annotation_to_camera_transformation[0, 0],
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

            # TODO debug
            # ann = coco_format['annotations'][-1]
            # pt_3d = unproject_2d_to_3d(
            #     ann["amodel_center"], ann["depth"], np.asarray(calib)
            # )
            # # pt_3d[1] += box_dimensions[2] / 2
            # print("location", ann["location"])
            # print("loc model", pt_3d)
            # print()
            # TODO end debug

    # TODO more than one video/scene
    num_video += 1
    coco_format["videos"].append({"id": num_video, "file_name": f'scene-{num_video:04d}'})

    out_path = pixset_annotations_path + "train.json"
    print("out_path", out_path)
    json.dump(coco_format, open(out_path, "w")) # TODO uncomment

    bar.finish()

