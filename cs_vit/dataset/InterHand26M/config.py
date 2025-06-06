# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import sys
import numpy as np

class Config:

    ## dataset (InterHand26M, MSCOCO, ReInterHand, HIC)
    trainset_3d = ['InterHand26M', 'ReInterHand']
    trainset_2d = ['MSCOCO']
    testset = 'HIC'

    ## model setting
    body_resnet_type = 50
    hand_resnet_type = 50
    trans_resnet_type = 18

    ## input, output
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    input_hand_shape = (256, 256)
    input_hm_shape = (64, 64, 64)
    output_body_hm_shape = (8, 64, 48)
    output_hand_hm_shape = (8, 8, 8)
    focal = (5000, 5000) # virtual focal lengths
    princpt = (input_hand_shape[1]/2, input_hand_shape[0]/2) # virtual principal point position
    bbox_3d_size = 0.3
    camera_3d_size = 0.6
    sigma = 2.5

    ## training config
    lr = 1e-4
    lr_dec_factor = 10
    lr_dec_epoch = [4]
    end_epoch = 7
    train_batch_size = 128

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 16
    num_gpus = 1
    continue_train = False

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..', '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'smplx_models')

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

# sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
# from .utils.dir import add_pypath, make_folder
# add_pypath(osp.join(cfg.data_dir))
# for i in range(len(cfg.trainset_3d)):
#     add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
# for i in range(len(cfg.trainset_2d)):
#     add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
# add_pypath(osp.join(cfg.data_dir, cfg.testset))
# make_folder(cfg.model_dir)
# make_folder(cfg.vis_dir)
# make_folder(cfg.log_dir)
# make_folder(cfg.result_dir)