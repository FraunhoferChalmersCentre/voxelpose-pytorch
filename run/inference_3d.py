import torch
import torchvision.transforms as transforms
import argparse
import os
import json
import numpy as np

from lib.core.config import config, update_config
from lib.utils.utils import create_logger
from lib.dataset import get_dataset
from lib.models.multi_person_posenet import get_multi_person_pose_net
import lib.dataset as dataset
import lib.models as models
from lib.utils.heatmaps import save_interactive_3d_skeleton_with_slider


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


def run_inference(config, model, dataloader, output_dir):
    """
    Runs inference on the dataset and saves predictions.
    """
    model.eval()  # Set the model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it does not exist

    preds = []
    metas = []
    with torch.no_grad():  # Disable gradient computation for inference
        for i, (inputs, meta) in enumerate(dataloader):
            pred, heatmaps, grid_centers, _, _, _ = model(views=inputs, meta=meta)
            pred = pred.detach().cpu()
            metas.append(meta)
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    save_interactive_3d_skeleton_with_slider(preds, output_path="/home/anders.sjoberg/projects/pose-estimation/external/voxelpose/output/hm_html/my_skeletons.html", meta=meta)

    print(f"✅ Inference completed. Results saved to {output_dir}")


def main():
    """
    Main function for running inference.
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
