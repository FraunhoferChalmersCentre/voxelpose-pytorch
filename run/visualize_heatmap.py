"""
Heatmap Visualization Script for VoxelPose

This script runs inference on a multi-view pose estimation model and visualizes
the predicted heatmaps overlaid on input images. It supports batch processing,
multi-GPU inference, and saves results in a structured directory format. The script
also triangulates 2D joint detections to 3D coordinates and exports joint data.

Features:
- Loads experiment configuration and trained model
- Runs inference on a test dataset using multiple GPUs
- Overlays predicted heatmaps on original images for each joint and view
- Triangulates multi-view 2D joint detections to 3D coordinates
- Saves visualizations and joint data to output directory
- Exports all joint results to Excel for further analysis

Example Usage:
    python external/voxelpose/run/visualize_heatmap.py \
        --cfg configs/panoptic/resnet50/inference_prn64_cpn80x80x20_960x512_cam5.yaml \
        --output-dir output/heatmaps

Requirements:
- Trained VoxelPose model and configuration file
- Test dataset prepared in the expected format
- CUDA-enabled GPUs for inference

"""
import torch
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from collections import defaultdict

from lib.core.config import config, update_config
from lib.utils.utils import create_logger
import lib.dataset as dataset
import lib.models as models
from lib.utils.definitions import *


def parse_args():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description="Run inference on a dataset")
    parser.add_argument('--cfg', help='Experiment configuration file', required=True, type=str)
    parser.add_argument('--output-dir', help='Directory to save inference results', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def overlay_heatmap(image, heatmap):
    """
    Overlay a heatmap onto an image for visualization.

    Args:
        image (np.ndarray): The original image (H x W x 3).
        heatmap (np.ndarray): The heatmap to overlay (H x W).
        alpha (float): Transparency factor for the heatmap overlay.

    Returns:
        np.ndarray: Image with heatmap overlay.
    """
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Convert grayscale heatmap to color
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)  # Blend images
    return overlay


def get_center(heatmap):
    """
    Find the coordinates of the maximum value in a heatmap.

    Args:
        heatmap (np.ndarray): The heatmap array.

    Returns:
        tuple: (x, y) coordinates of the maximum value.
    """
    if heatmap.max()<0.5:
        return np.array([np.nan,np.nan])
    # Find maximum value coordinates
    max_val = heatmap.max()
    max_locs = np.argwhere(heatmap == max_val)
    center = max_locs.mean(axis=0)  # [y, x] as floats

    return center[::-1]


def plot_3d_joints(points_3d, joint_names=None, limbs=None, title="3D pose", elev=15, azim=-70):
    """
    Plots 3D joints with optional joint names and skeleton limbs.

    :param points_3d: np.array of shape (N, 3) or (3, N)
    :param joint_names: dict mapping index -> name
    :param limbs: list of pairs [(i1, i2), ...] defining bones
    :param title: plot title
    :param elev: elevation angle for 3D view
    :param azim: azimuth angle for 3D view
    """

    if joint_names is None:
        joint_names = JOINTS_DEF_INV

    if limbs is None:
        limbs = LIMBS

    if points_3d.shape[0] == 3 and points_3d.shape[1] != 3:
        points_3d = points_3d.T

    xs, ys, zs = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='b', marker='o')

    for i in range(points_3d.shape[0]):
        name = joint_names.get(i, str(i))
        ax.text(xs[i], ys[i], zs[i], name, size=8)

    if limbs:
        for i1, i2 in limbs:
            xline = [xs[i1], xs[i2]]
            yline = [ys[i1], ys[i2]]
            zline = [zs[i1], zs[i2]]
            ax.plot(xline, yline, zline, 'r')

    ax.set_xlabel('X (right)')
    ax.set_ylabel('Y (forward)')
    ax.set_zlabel('Z (up)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()




def triangulate_pts_adaptive(joint_data, metas, img_idx=0):
    """
    Triangulate 3-D joint locations from multi-view 2-D detections.

    Args
    ----
    joint_data : list(dict)
        Each dict has keys ['x', 'y', 'joint_idx', 'view_idx', 'image'].
    metas : list(dict)
        Per-view metadata; metas[v]['camera'] must contain 'K', 'R', 'T'.
    img_idx : int
        Which image slice to take intrinsics / extrinsics from.

    Returns
    -------
    np.ndarray
        (num_joints, 3) array of 3-D points in classical Z-up coords.
        Missing / untriangulated joints are NaN.
    """
    # 1) pandas pivot: rows = (image, joint_idx)  cols = (coord, view_idx)
    df = pd.DataFrame(joint_data)
    df_pivot = df.pivot_table(index=['image', 'joint_idx'],
                              columns='view_idx',
                              values=['x', 'y'])

    num_joints  = df['joint_idx'].nunique()
    view_indices = df['view_idx'].unique()
    # Choose the smallest view index as reference
    ref_view_idx = view_indices.min()

    # 2) Pre-compute projection matrices for every camera
    proj_mats = {}
    R_ref_classic = np.array(metas[ref_view_idx]['camera']['R'][img_idx])
    t_ref_classic = np.array(metas[ref_view_idx]['camera']['T'][img_idx]).reshape(3, 1)

    R_ref_cv = R_ref_classic @ M.T  # classic  → OpenCV
    t_ref_cv = -M.T @ R_ref_classic.T @ t_ref_classic

    for v_idx in view_indices:
        # --- Intrinsics ----------------------------------------------------------
        K = np.array(metas[v_idx]['camera']['K'][img_idx])

        # --- Extrinsics in classic camera coords ---------------------------------
        R_classic = np.array(metas[v_idx]['camera']['R'][img_idx])
        t_classic = np.array(metas[v_idx]['camera']['T'][img_idx]).reshape(3, 1)

        # --- Convert to OpenCV axis convention -----------------------------------
        R_cv = R_classic @ M.T
        t_cv = -M.T @ R_classic.T @ t_classic

        # --- Relative pose w.r.t. reference camera in OpenCV coords --------------
        if v_idx == ref_view_idx:
            R_rel = np.eye(3)
            t_rel = np.zeros((3, 1))
        else:
            R_rel = R_cv @ R_ref_cv.T
            t_rel = t_cv - R_rel @ t_ref_cv

        # --- Final 3×4 projection matrix -----------------------------------------
        proj_mats[v_idx] = K @ np.hstack((R_rel, t_rel))

    # 3) Triangulate joint-by-joint
    joint_results = np.full((num_joints, 3), np.nan)

    for (_, j_idx), row in df_pivot.iterrows():
        # --- collect all valid (x,y,P) for this joint ----------------------
        pts2d, Ps = [], []
        for v_idx in view_indices:
            if not pd.isna(row.get(('x', v_idx), np.nan)):
                x = row[('x', v_idx)]
                y = row[('y', v_idx)]
                pts2d.append([x, y])
                Ps.append(proj_mats[v_idx])

        m = len(Ps)           # how many views for this joint?
        if m < 2:
            # < 2 views → cannot triangulate
            continue

        pts2d = np.asarray(pts2d, dtype=np.float64)

        # --- exactly two views --------------------------------------------
        if m == 2:
            P1, P2 = Ps
            pt1 = pts2d[0].reshape(2, 1)
            pt2 = pts2d[1].reshape(2, 1)

            X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)   # (4,1)
            X   = (X_h[:3] / X_h[3]).ravel()

        # --- three or more views ------------------------------------------
        else:
            A_rows = []
            for (x, y), P in zip(pts2d, Ps):
                A_rows.append(x * P[2] - P[0])
                A_rows.append(y * P[2] - P[1])
            A = np.vstack(A_rows)                           # (2m, 4)
            _, _, Vt = np.linalg.svd(A)
            X_h = Vt[-1]
            X   = X_h[:3] / X_h[3]

        # Classical Z-up coordinates
        joint_results[j_idx] = X @ M

    return joint_results



def visualize_heatmaps(model, dataloader, output_dir):
    """
    Run inference, overlay heatmaps, save images, and export joint data.

    Args:
        model (torch.nn.Module): Trained pose estimation model.
        dataloader (torch.utils.data.DataLoader): DataLoader for test images.
        output_dir (str): Directory to save visualizations and joint data.
        meta (dict, optional): Additional metadata for visualization.

    Returns:
        None
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    image_counter = 0  # Counter for numbering images uniquely

    all_joint_rows = []
    with torch.no_grad():
        for views, metas in dataloader:
            all_heatmaps = []

            for view_idx, view in enumerate(views):
                view = view.cuda()
                heatmaps = model.module.backbone(view)
                heatmaps_np = heatmaps.cpu().numpy()
                all_heatmaps.append(heatmaps_np)

            batch_size = heatmaps.shape[0]
            num_joints = heatmaps.shape[1]

            for img_idx in range(batch_size):
                image_folder = os.path.join(output_dir, f"images/image_{image_counter:04d}")
                os.makedirs(image_folder, exist_ok=True)
                joints_data = []

                for view_idx, heatmaps in enumerate(all_heatmaps):
                    view_folder = os.path.join(image_folder, f"view_{view_idx}")
                    os.makedirs(view_folder, exist_ok=True)

                    original_img_path = metas[view_idx]['image'][img_idx]
                    original_img = cv2.imread(original_img_path)
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                    for joint_idx in range(num_joints):
                        heatmap = heatmaps[img_idx, joint_idx, :, :]
                        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

                        # Get heatmap center
                        joint_pts = get_center(heatmap)
                        joints_data.append({'x':joint_pts[0],
                                            'y':joint_pts[1],
                                            'joint_idx':joint_idx,
                                            'view_idx':view_idx,
                                            'image':img_idx,
                                            'img_path':original_img_path})

                        # Normalize heatmaps for overlay
                        heatmap = np.uint8(255 * heatmap / np.max(heatmap))

                        # Overlay heatmap
                        overlayed_img = overlay_heatmap(original_img, heatmap)

                        # Get joint name
                        joint_name = JOINTS_DEF_INV.get(joint_idx, f"joint_{joint_idx}")

                        # Save image
                        output_file = os.path.join(view_folder, f"{joint_name}.jpg")
                        #plt.imsave(output_file, overlayed_img)

                # Triangulate heatmap joints
                triangulated_pts = triangulate_pts_adaptive(joint_data=joints_data, metas=metas)

                for batch_idx, joint_dict in enumerate(joints_data):
                    current_joint = joint_dict['joint_idx']
                    xyz = triangulated_pts[current_joint,:]
                    row = {
                        'image': joint_dict['image'],
                        'joint_idx': joint_dict['joint_idx'],
                        'view_idx': joint_dict['view_idx'],
                        'img_path': joint_dict['img_path'],
                        'x_2d': joint_dict['x'],
                        'y_2d': joint_dict['y'],
                        'X_3d': xyz[0],
                        'Y_3d': xyz[1],
                        'Z_3d': xyz[2],
                    }
                    all_joint_rows.append(row)


                print(f"✅ Saved heatmaps for image {image_counter}, from {len(views)} views")

                image_counter += 1  # Increment image counter

    print(f"✅ Heatmap visualization completed. Results saved to {output_dir}")
    df_all_joints = pd.DataFrame(all_joint_rows)
    df_all_joints.to_excel(os.path.join(output_dir, "all_joints.xlsx"))



def main():
    """
    Main function for running heatmap visualization.
    """
    args = parse_args()

    # Create logger
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'inference')
    logger.info(f"Running inference with config: {args.cfg}")

    # Setup GPU processing
    gpus = [int(i) for i in config.GPUS.split(',')]
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # Load dataset
    print("=> Loading dataset ..")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_dataset = eval(f'dataset.{config.DATASET.TEST_DATASET}')(
        config, None, False,
        transforms.Compose([transforms.ToTensor(), normalize])
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load the trained model
    print('=> Loading model ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info(f"=> Load model state {test_model_file}")
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    # Run heatmap visualization
    visualize_heatmaps(model, test_loader, args.output_dir)


if __name__ == '__main__':
    main()
