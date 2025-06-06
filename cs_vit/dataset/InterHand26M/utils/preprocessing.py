# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import cv2
import random
from ..config import cfg
import math
# from .mano import mano
from ....utils.mano import mano
from .transforms import cam2pixel, transform_joint_to_other_db
from plyfile import PlyData, PlyElement
import torch
import kornia


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()

    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1]
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img)

    x_center = (xmin+xmax)/2.; width = xmax-xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio

    y_center = (ymin+ymax)/2.; height = ymax-ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def crop_img(img: torch.Tensor, bbox_center, bbox_size, squarify=True, avoid_zero=False):
    '''
    center, size 均遵从 interwild 工作中的定义，是一个二元组。第一、二个元素分别
    是水平方向和垂直方向的位置和长度。
    '''
    assert isinstance(img, torch.Tensor), "Only torch.Tensor image is supported"

    w_center, h_center = bbox_center
    width, height = bbox_size

    if squarify:
        length = max(width, height)
        width = length
        height = length
    if avoid_zero:
        width = max(width, 2)  # ! use 2 instead of 1
        height = max(height, 2)

    w_min = (w_center - width / 2)
    h_min = (h_center - height / 2)
    w_max = (w_center + width / 2)
    h_max = (h_center + height / 2)
    boxes = torch.tensor([[
        [w_min, h_min], [w_max, h_min], [w_max, h_max], [w_min, h_max]
    ]])
    output_size = (int(height), int(width))

    cropped_img = kornia.geometry.transform.crop_and_resize(img[None,...], boxes, output_size)
    return cropped_img[0]

def process_bbox(bbox, img_width, img_height, do_sanitize=True, extend_ratio=1.25):
    if do_sanitize:
        bbox = sanitize_bbox(bbox, img_width, img_height)
        if bbox is None:
            return bbox

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*extend_ratio
    bbox[3] = h*extend_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    bbox = bbox.astype(np.float32)
    return bbox

def get_aug_config():
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip

def augmentation(img, bbox, data_split, enforce_flip=None):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config()
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False

    if enforce_flip is None:
        pass
    elif enforce_flip is True:
        do_flip = True
    elif enforce_flip is False:
        do_flip = False

    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, trans, inv_trans, rot, do_flip

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def distort_projection_fisheye(point, focal, princpt, D):
    z = point[:,:,2].clone()

    # distort
    point_ndc = point[:,:,:2] / z[:,:,None]
    r = torch.sqrt(torch.sum(point_ndc ** 2, 2))
    theta = torch.atan(r)
    theta_d = theta * (
            1
            + D[:,None,0] * theta.pow(2)
            + D[:,None,1] * theta.pow(4)
            + D[:,None,2] * theta.pow(6)
            + D[:,None,3] * theta.pow(8)
    )
    point_ndc = point_ndc * (theta_d / r)[:,:,None]

    # project
    x = point_ndc[:,:,0]
    y = point_ndc[:,:,1]
    x = x * focal[:,None,0] + princpt[:,None,0]
    y = y * focal[:,None,1] + princpt[:,None,1]
    point_proj = torch.stack((x,y,z),2)
    return point_proj

def transform_db_data(joint_img, joint_cam, joint_valid, rel_trans, do_flip, img_shape, flip_pairs, img2bb_trans, rot, src_joints_name, target_joints_name):
    joint_img, joint_cam, joint_valid, rel_trans = joint_img.copy(), joint_cam.copy(), joint_valid.copy(), rel_trans.copy()

    # flip augmentation
    if do_flip:
        joint_cam[:,0] = -joint_cam[:,0]
        joint_img[:,0] = img_shape[1] - 1 - joint_img[:,0]
        rel_trans[1:3] = -rel_trans[1:3]
        for pair in flip_pairs:
            joint_img[pair[0],:], joint_img[pair[1],:] = joint_img[pair[1],:].copy(), joint_img[pair[0],:].copy()
            joint_cam[pair[0],:], joint_cam[pair[1],:] = joint_cam[pair[1],:].copy(), joint_cam[pair[0],:].copy()
            joint_valid[pair[0],:], joint_valid[pair[1],:] = joint_valid[pair[1],:].copy(), joint_valid[pair[0],:].copy()

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1,0)).transpose(1,0)
    rel_trans = np.dot(rot_aug_mat, rel_trans[:,None]).reshape(3)

    # affine transformation and root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
    joint_img[:,:2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
    joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_body_hm_shape[2]
    joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_body_hm_shape[1]
    joint_img[:,2] = (joint_img[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_body_hm_shape[0]

    # check truncation
    joint_trunc = joint_valid * ((joint_img[:,0] >= 0) * (joint_img[:,0] < cfg.output_body_hm_shape[2]) * \
                (joint_img[:,1] >= 0) * (joint_img[:,1] < cfg.output_body_hm_shape[1]) * \
                (joint_img[:,2] >= 0) * (joint_img[:,2] < cfg.output_body_hm_shape[0])).reshape(-1,1).astype(np.float32)

    # change joint order
    joint_img = transform_joint_to_other_db(joint_img, src_joints_name, target_joints_name)
    joint_cam = transform_joint_to_other_db(joint_cam, src_joints_name, target_joints_name)
    joint_valid = transform_joint_to_other_db(joint_valid, src_joints_name, target_joints_name)
    joint_trunc = transform_joint_to_other_db(joint_trunc, src_joints_name, target_joints_name)
    return joint_img, joint_cam, joint_valid, joint_trunc, rel_trans

def transform_mano_data(joint_img, joint_cam, mesh_cam, joint_valid, rel_trans, pose, img2bb_trans, rot):
    joint_img, pose = joint_img.copy(), pose.copy()

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    mesh_cam = np.dot(rot_aug_mat, mesh_cam.transpose(1,0)).transpose(1,0)
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1,0)).transpose(1,0)
    rel_trans = np.dot(rot_aug_mat, rel_trans[:,None]).reshape(3)
    pose = pose.reshape(-1,3)
    for h in ('right', 'left'):
        if h == 'right':
            root_joint_idx = mano.orig_root_joint_idx
        else:
            root_joint_idx = mano.orig_root_joint_idx + mano.orig_joint_num
        root_pose = pose[root_joint_idx,:]
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        pose[root_joint_idx] = root_pose.reshape(3)
    pose = pose.reshape(-1)

    # affine transformation and root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
    joint_img[:,:2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
    joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[1] * cfg.output_body_hm_shape[2]
    joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0] * cfg.output_body_hm_shape[1]
    joint_img[:,2] = (joint_img[:,2] / (cfg.bbox_3d_size / 2) + 1)/2. * cfg.output_body_hm_shape[0]

    # check truncation
    joint_trunc = joint_valid * ((joint_img[:,0] >= 0) * (joint_img[:,0] < cfg.output_body_hm_shape[2]) * \
                (joint_img[:,1] >= 0) * (joint_img[:,1] < cfg.output_body_hm_shape[1]) * \
                (joint_img[:,2] >= 0) * (joint_img[:,2] < cfg.output_body_hm_shape[0])).reshape(-1,1).astype(np.float32)

    return joint_img, joint_cam, mesh_cam, joint_trunc, rel_trans, pose

def get_mano_data(mano_param, cam_param, do_flip, img_shape):
    pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
    hand_type = mano_param['hand_type']
    pose = torch.FloatTensor(pose).view(-1,3); shape = torch.FloatTensor(shape).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1,-1) # translation vector
    if do_flip:
        if hand_type == 'right':
            hand_type = 'left'
        else:
            hand_type = 'right'

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        root_pose = pose[mano.orig_root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        pose[mano.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in mano.orig_flip_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
        trans[:,0] *= -1 # multiply -1

    # get root joint coordinate
    root_pose = pose[mano.orig_root_joint_idx].view(1,3)
    hand_pose = torch.cat((pose[:mano.orig_root_joint_idx,:], pose[mano.orig_root_joint_idx+1:,:])).view(1,-1)
    with torch.no_grad():
        output = mano.layer[hand_type](betas=shape, hand_pose=hand_pose, global_orient=root_pose, transl=trans)
    mesh_coord = output.vertices[0].numpy()
    joint_coord = np.dot(mano.sh_joint_regressor, mesh_coord)

    # bring geometry to the original (before flip) position
    if do_flip:
        flip_trans_x = joint_coord[mano.sh_root_joint_idx,0] * -2
        mesh_coord[:,0] += flip_trans_x
        joint_coord[:,0] += flip_trans_x

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
        root_coord = joint_coord[mano.sh_root_joint_idx,None,:].copy()
        joint_coord = joint_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t
        mesh_coord = mesh_coord - root_coord + np.dot(R, root_coord.transpose(1,0)).transpose(1,0) + t

    # flip translation
    if do_flip: # avg of old and new root joint should be image center.
        focal, princpt = cam_param['focal'], cam_param['princpt']
        flip_trans_x = 2 * (((img_shape[1] - 1)/2. - princpt[0]) / focal[0] * joint_coord[mano.sh_root_joint_idx,2]) - 2 * joint_coord[mano.sh_root_joint_idx][0]
        mesh_coord[:,0] += flip_trans_x
        joint_coord[:,0] += flip_trans_x

    # image projection
    mesh_cam = mesh_coord # camera-centered 3D coordinates (not root-relative)
    joint_cam = joint_coord # camera-centered 3D coordinates (not root-relative)
    if 'D' in cam_param:
        joint_img = distort_projection_fisheye(torch.from_numpy(joint_cam)[None], torch.from_numpy(cam_param['focal'])[None], torch.from_numpy(cam_param['princpt'])[None], torch.from_numpy(cam_param['D'])[None])
        joint_img = joint_img[0].numpy()
    else:
        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
    joint_img = joint_img[:,:2]

    pose = pose.numpy().reshape(-1)
    shape = shape.numpy().reshape(-1)
    return joint_img, joint_cam, mesh_cam, pose, shape

def get_iou(box1, box2, form):
    box1 = box1.copy()
    box2 = box2.copy()
    box1 = box1.reshape(-1,4)
    box2 = box2.reshape(-1,4)

    if form == 'xyxy':
        pass
    elif form == 'xywh':
        box1[:,2:] += box1[:,:2] # xywh -> xyxy
        box2[:,2:] += box2[:,:2] # xywh -> xyxy

    xmin = np.maximum(box1[:,0], box2[:,0])
    ymin = np.maximum(box1[:,1], box2[:,1])
    xmax = np.minimum(box1[:,2], box2[:,2])
    ymax = np.minimum(box1[:,3], box2[:,3])
    inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box1_area = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    box2_area = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-5)
    return iou

def load_obj(file_name):
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x,y,z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x,y,z]))
    return np.stack(v)

def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x,y,z),1)
    return v