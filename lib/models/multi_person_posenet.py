# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from typing import List, Dict

from lib.models import pose_resnet
from lib.models.cuboid_proposal_net import CuboidProposalNet
from lib.models.pose_regression_net import PoseRegressionNet
from lib.core.loss import PerJointMSELoss
from lib.core.loss import PerJointL1Loss
from lib.utils.heatmaps import visualize_heatmaps

class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.root_net = CuboidProposalNet(cfg)
        self.pose_net = PoseRegressionNet(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET

    def forward(self,
                views: List[torch.Tensor] = None,
                meta: List[Dict] = None,
                targets_2d: List[torch.Tensor] = None,
                weights_2d: List[torch.Tensor] = None,
                targets_3d: torch.Tensor = None,
                input_heatmaps: List[torch.Tensor] = None):
        """
        Perform forward-pass for the network.

        Args:
            views (list[torch.Tensor]): A list of input image tensor of shape (B, C, H, W),
                where B is the batch size, C is the number of channels in this case 3 RGB,
                H is the height, and W is the width.
                Each element in the list is from different cameras.
            meta (list[dict]): Metadata associated with the inputs, containing information:
                - 'image': Path to images in views.
                - 'num_person': Number of people in each sample.
                - 'roots_3d': 3D coordinates of root joints (used if USE_GT is True).
                - 'joints_3d': 3D joint positions (for calculating pose loss).
                - 'joints_3d_vis': Visibility masks for 3D joints.
                - 'center: Center of the image for affine transformations, shape (2,).
                    Typically set to the midpoint of the image dimensions (e.g., [960, 540] for a 1920x1080 image).
                - 'scale': Scaling factor used to resize the image to the target input size of the network.
                - 'rotation': Rotation angle (in degrees) applied to the image during augmentation or preprocessing.
                - 'camera' (dict): Camera calibration parameters for projecting 3D points to the 2D image plane:
                    - 'R': Rotation matrix.
                    - 'T': Translation vector.
                    - 'fx', 'fy': Focal lengths along the x and y axes, respectively.
                    - 'cx', 'cy': Principal point offsets along the x and y axes, respectively.
                    - 'k': Radial distortion coefficients.
                    - 'p': Tangential distortion coefficient.
            targets_2d (list[torch.Tensor]): Ground truth 2D heatmaps of shape (B, J, H', W'),
                where J is the number of joints, H' and W' are heatmap dimensions.
                Each element in the list is from different cameras.
            weights_2d (list[torch.Tensor]): Weights for the 2D heatmaps, typically of shape (B, J, 1).
                Each element in the list is from different cameras.
            targets_3d (torch.Tensor): Ground truth 3D heatmaps of shape (B, D, H', W'),
                where D is the depth dimension.
            input_heatmaps (list[torch.Tensor]): Precomputed input heatmaps, used as an alternative
                to views if provided.

        Returns:
            pred (torch.Tensor): Predicted joint locations and confidences of shape (B, N, J, 5),
                where B is batch-size, N is the number of proposals (pre-defined maximum in config),
                J is the number of joints, and the last dimension contains:
                - [x, y, z] real coordinates [index: 0 to 2] (3 elements).
                - Presence of joint: 0 = present, -1 = no present,
                i.e., confidence score > threshold [index: 3] (1 element).
                - Confidence score of joint [index: 4] (1 element).
            all_heatmaps (list[torch.Tensor]): List of 2D heatmaps for each view.
            grid_centers (torch.Tensor): Anchor locations and their associated information of shape (B, N, 5),
                where B is batch-size, N is the number of proposals (pre-defined maximum in config),
                and the last dimension contains:
                - [x, y, z] real coordinates of anchor point [index: 0 to 2] (3 elements).
                - Presence of person (root joint?): 0 = present, -1 = no present,
                i.e., confidence score > threshold [index: 3] (1 element).
                - Confidence score of person (root joint?) [index: 4] (1 element).
            loss_2d (torch.Tensor): 2D heatmap loss, computed as mean squared error.
            loss_3d (torch.Tensor): 3D heatmap loss, computed as mean squared error.
            loss_cord (torch.Tensor): Joint coordinate regression loss, computed as L1 loss.
        """

        if views is not None:
            all_heatmaps = []
            for view in views:
                heatmaps = self.backbone(view)
                all_heatmaps.append(heatmaps)
        else:
            all_heatmaps = input_heatmaps

        # visualize_heatmaps(views, all_heatmaps,'/home/anders.sjoberg/projects/pose-estimation/external/voxelpose/output/heatmapsdebug', meta=meta, heatmapsize=False) # TODO

        # all_heatmaps = targets_2d
        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]

        # calculate 2D heatmap loss
        criterion = PerJointMSELoss().cuda()
        loss_2d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        if targets_2d is not None:
            for t, w, o in zip(targets_2d, weights_2d, all_heatmaps):
                loss_2d += criterion(o, t, True, w)
            loss_2d /= len(all_heatmaps)

        loss_3d = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        if self.USE_GT:
            num_person = meta[0]['num_person']
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]['roots_3d'].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, :num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, :num_person[i], 4] = 1.0
        else:
            root_cubes, grid_centers = self.root_net(all_heatmaps, meta)

            # calculate 3D heatmap loss
            if targets_3d is not None:
                loss_3d = criterion(root_cubes, targets_3d)
            del root_cubes

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt

        loss_cord = criterion(torch.zeros(1, device=device), torch.zeros(1, device=device))
        criterion_cord = PerJointL1Loss().cuda()
        count = 0

        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:
                single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n])
                pred[:, n, :, 0:3] = single_pose.detach()

                # calculate 3D pose loss
                if self.training and 'joints_3d' in meta[0] and 'joints_3d_vis' in meta[0]:
                    gt_3d = meta[0]['joints_3d'].float()
                    for i in range(batch_size):
                        if pred[i, n, 0, 3] >= 0:
                            targets = gt_3d[i:i + 1, pred[i, n, 0, 3].long()]
                            weights_3d = meta[0]['joints_3d_vis'][i:i + 1, pred[i, n, 0, 3].long(), :, 0:1].float()
                            count += 1
                            loss_cord = (loss_cord * (count - 1) +
                                         criterion_cord(single_pose[i:i + 1], targets, True, weights_3d)) / count
                del single_pose

        return pred, all_heatmaps, grid_centers, loss_2d, loss_3d, loss_cord


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
