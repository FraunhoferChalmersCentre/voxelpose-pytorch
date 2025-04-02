import torch
import torchvision.transforms as transforms
import argparse
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

                for view_idx, heatmaps in enumerate(all_heatmaps):
                    view_folder = os.path.join(image_folder, f"view_{view_idx}")
                    os.makedirs(view_folder, exist_ok=True)

                    original_img_path = metas[view_idx]['image'][img_idx]
                    original_img = cv2.imread(original_img_path)
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                    for joint_idx in range(num_joints):
                        heatmap = heatmaps[img_idx, joint_idx, :, :]
                        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                        heatmap = np.uint8(255 * heatmap / np.max(heatmap))

                        # Overlay heatmap
                        overlayed_img = overlay_heatmap(original_img, heatmap)

                        # Get joint name
                        joint_name = JOINTS_DEF.get(joint_idx, f"joint_{joint_idx}")

                        # Save image
                        output_file = os.path.join(view_folder, f"{joint_name}.jpg")
                        plt.imsave(output_file, overlayed_img)

                print(f"✅ Saved heatmaps for image {image_counter}, from {len(views)} views")

                image_counter += 1  # Increment image counter

    print(f"✅ Heatmap visualization completed. Results saved to {output_dir}")


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
