"""
Multi-View Pose Inference Script for VoxelPose

This script runs inference using a trained VoxelPose model on a multi-view dataset.
It loads the experiment configuration, model weights, and test data, then predicts
3D poses and saves interactive visualizations.

Features:
- Loads experiment configuration and trained model checkpoint
- Runs inference on a multi-view test dataset using multiple GPUs
- Predicts 3D joint positions and heatmaps for each batch
- Triangulates joints from multi-view heatmaps to obtain 3D coordinates
- Saves interactive 3D skeleton visualizations for both model predictions and triangulated results

Example Usage:
    python external/voxelpose/run/inference_3d.py \
        --cfg configs/panoptic/resnet50/inference_prn64_cpn80x80x20_960x512_cam5.yaml \
        --output-dir output/inference_results

Requirements:
- Trained VoxelPose model and configuration file
- Test dataset prepared in the expected format
- CUDA-enabled GPUs for inference

"""
import torch
import torchvision.transforms as transforms
import argparse
import os

from lib.core.config import config, update_config
from lib.utils.utils import create_logger
from lib.dataset import get_dataset
from lib.models.multi_person_posenet import get_multi_person_pose_net
import lib.dataset as dataset
import lib.models as models
from lib.utils.heatmaps import save_interactive_3d_skeleton_with_slider
from lib.utils.triangulate_3d import triangulate_from_heatmaps


def parse_args():
    """
    Parse command-line arguments for inference.

    Returns:
        argparse.Namespace: Parsed arguments including configuration file path and output directory.
    """
    parser = argparse.ArgumentParser(description="Run inference on a dataset")
    parser.add_argument('--cfg', help='Experiment configuration file', required=True, type=str)
    parser.add_argument('--output-dir', help='Directory to save inference results', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def run_inference(config, model, dataloader, output_dir):
    """
    Runs inference on the dataset and saves predictions and visualizations.

    Args:
        config: Experiment configuration object.
        model: Trained VoxelPose model.
        dataloader: DataLoader for the test dataset.
        output_dir (str): Directory to save results.

    Workflow:
        - Sets model to evaluation mode
        - Iterates over batches, runs model to get predictions and heatmaps
        - Triangulates joints from multi-view heatmaps
        - Saves interactive 3D skeleton visualizations for both model predictions and triangulated joints
        - Prints completion message
    """
    model.eval()  # Set the model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it does not exist

    preds = []
    metas = []
    all_heatmaps = []
    with torch.no_grad():

            
        for i, (views, meta) in enumerate(dataloader):
            pred, heatmaps, grid_centers, _, _, _ = model(views=views, meta=meta)
            pred = pred.detach().cpu()

            heatmaps_np = [hm.detach().cpu().numpy() for hm in heatmaps]
            all_heatmaps.append(heatmaps_np)

            metas.append(meta)
            preds.append(pred)
            
    
    # Triangulate joints from heatmaps
    triangulated = triangulate_from_heatmaps(all_heatmaps, metas)
    preds = torch.cat(preds, dim=0)
    #triangulated_joints = np.concatenate(triangulated, axis=0)

    # Save both model predictions and triangulated results
    # For model predictions (shape: B, N, J, 5)
    save_interactive_3d_skeleton_with_slider(
        preds, 
        output_path=os.path.join(output_dir, "model_predictions.html"),
        meta=meta, 
        is_triangulated=False
    )

    # For triangulated points (shape: B, J, 3)
    save_interactive_3d_skeleton_with_slider(
        triangulated, 
        output_path=os.path.join(output_dir, "triangulated_joints.html"),
        meta=meta,
        is_triangulated=True
    )
    print(f"✅ Inference completed. Results saved to {output_dir}")


def main():
    """
    Main entry point for running multi-view pose inference.

    Workflow:
        - Parses command-line arguments
        - Sets up logger and GPU configuration
        - Loads test dataset and model
        - Loads model weights from checkpoint
        - Runs inference and saves results
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
    test_dataset = get_dataset(config.DATASET.TEST_DATASET)(
        config, None, False,
        transforms.Compose([transforms.ToTensor(), normalize])
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=0,#config.WORKERS,
        pin_memory=True
    )

    # Load the trained model
    print('=> Loading models ..')
    model = get_multi_person_pose_net(config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    # Run inference
    run_inference(config, model, test_loader, args.output_dir)


if __name__ == '__main__':
    main()
