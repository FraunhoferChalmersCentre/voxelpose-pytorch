import torch
import torchvision.transforms as transforms
import argparse
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

from lib.core.config import config, update_config
from lib.utils.utils import create_logger
import lib.dataset as dataset
import lib.models as models


JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}


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
    Overlay a heatmap onto an image.
    :param image: Original image (numpy array).
    :param heatmap: Heatmap (numpy array, same size as image).
    :return: Image with heatmap overlay.
    """
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Convert grayscale heatmap to color
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)  # Blend images
    return overlay

def get_center(heatmap):

    if heatmap.max()<0.5:
        return np.array([np.nan,np.nan])
    # Find maximum value coordinates
    max_val = heatmap.max()
    max_locs = np.argwhere(heatmap == max_val)
    center = max_locs.mean(axis=0)  # [y, x] as floats

    return center

def triangulate_pts(K1, K2, R, T, joint_data):

    # Filter data
    df_joint_pts = pd.DataFrame(joint_data)
    df_joints_sorted = df_joint_pts.sort_values(['image', 'joint_idx', 'view_idx'], ascending=[True, True, True])

    # Convert coordinate system to opencv
    M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    R =  R @ np.linalg.inv(M)

    # Get projection matrices to project image points to real world coordinates
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1)))) # For Camera 0 (reference)
    P2 = K2 @ np.hstack((R, T))

    P1 = P1.cpu().numpy() if hasattr(P1, 'cpu') else np.array(P1)
    P2 = P2.cpu().numpy() if hasattr(P2, 'cpu') else np.array(P2)

    # Set multi-index for alignment
    df0 = df_joints_sorted[df_joints_sorted['view_idx'] == 0].set_index(['image', 'joint_idx'])
    df1 = df_joints_sorted[df_joints_sorted['view_idx'] == 1].set_index(['image', 'joint_idx'])

    # Align on index and extract as arrays
    aligned0, aligned1 = df0.align(df1)

    xy_0 = aligned0[['x', 'y']].to_numpy()
    xy_1 = aligned1[['x', 'y']].to_numpy()

    xy_0 = xy_0.astype(np.float64).T  # shape (2, N)
    xy_1 = xy_1.astype(np.float64).T  # shape (2, N)

    #for image in iterrows(df_pts):
    # Triangulate to homogeneous coordinates
    points_4d = cv2.triangulatePoints(P1, P2, xy_0, xy_1)

    # Convert to 3D (non-homogeneous coordinates)
    points_3d = points_4d[:3, :] / points_4d[3, :]  # Shape (3, N)

    # from mpl_toolkits.mplot3d import Axes3D  # Not strictly needed in recent matplotlib, but safe to import
    # # If points_3d is shape (N, 3), transpose for plotting, or use .T if shape (3, N)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # If your points are shape (N, 3):
    # ax.scatter(points_3d[0, :], points_3d[1,:], points_3d[ 2,:], marker='.')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    return points_3d

def visualize_heatmaps(model, dataloader, output_dir):
    """
    Runs inference and overlays heatmaps onto images for multiple views.
    Saves results in structured directories: images -> views -> joints.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Joint index mapping for easy lookup
    JOINTS_DEF = {
        0: 'neck', 1: 'nose', 2: 'mid-hip', 3: 'l-shoulder', 4: 'l-elbow',
        5: 'l-wrist', 6: 'l-hip', 7: 'l-knee', 8: 'l-ankle', 9: 'r-shoulder',
        10: 'r-elbow', 11: 'r-wrist', 12: 'r-hip', 13: 'r-knee', 14: 'r-ankle'
    }

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
                                            'image':f"{image_counter:04d}",
                                            'img_path':original_img_path})

                        # Normalize heatmaps for overlay
                        heatmap = np.uint8(255 * heatmap / np.max(heatmap))

                        # Overlay heatmap
                        overlayed_img = overlay_heatmap(original_img, heatmap)

                        # Get joint name
                        joint_name = JOINTS_DEF.get(joint_idx, f"joint_{joint_idx}")

                        # Save image
                        output_file = os.path.join(view_folder, f"{joint_name}.jpg")
                        #plt.imsave(output_file, overlayed_img)

                # Triangulate heatmap joints
                try:
                    triangulated_pts = triangulate_pts(K1=metas[0]['camera']['K'][img_idx],
                                                             K2=metas[1]['camera']['K'][img_idx],
                                                             R=metas[0]['camera']['R'][img_idx],
                                                             T=metas[0]['camera']['T'][img_idx],
                                                             joint_data=joints_data)

                    for batch_idx, joint_dict in enumerate(joints_data):
                        current_joint = joint_dict['joint_idx']
                        xyz = triangulated_pts[:,current_joint]
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

                except Exception as e:
                    print(f"Views length: {len(views)}")

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
