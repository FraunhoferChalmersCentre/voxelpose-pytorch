import glob
import os.path as osp
import cv2
import json_tricks as json
from lib.dataset.JointsDataset import JointsDataset
from lib.utils.transforms import get_affine_transform, get_scale
from lib.utils.definitions import *


class PanopticInference(JointsDataset):
    def __init__(self, cfg, cam_list, seq_list, transform=None):
        """
        Initialize the dataset for inference.
        :param cfg: Configuration file.
        :param cam_list: List of camera (panel, node) pairs.
        :param seq_list: List of sequences (folders) to process.
        :param transform: Transformations to be applied to images.
        """
        super().__init__(cfg, 'inference', False, transform)
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.cam_list = cam_list if cam_list is not None else [(0, 0), (0, 1)]
        self.num_views = len(self.cam_list)
        self.sequence_list = cfg.DATASET.FOLDERS
        self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        """
        Creates a database containing all camera views for each image.
        """
        db = []
        for seq in self.sequence_list:
            cameras = self._get_cam(seq)
            image_dir = osp.join(self.dataset_root, seq, 'hdImgs')

            # Iterate over all selected cameras
            for (panel, node), cam_params in cameras.items():
                image_files = sorted(glob.glob(osp.join(image_dir, f'{panel:02d}_{node:02d}', '*.jpg')))

                for img_idx, image_file in enumerate(image_files):
                    # Store image paths and corresponding camera parameters
                    if len(db) <= img_idx:
                        db.append({'images': [], 'cameras': []})

                    # Copy camera parameters and use the correct T from _get_cam()
                    updated_cam_params = {
                        'K': cam_params['K'],
                        'distCoef': cam_params['distCoef'],
                        'R': cam_params['R'],
                        'T': -np.dot(cam_params['R'].T, cam_params['t']),  # Use T directly, without modifying it
                        'fx': cam_params['K'][0, 0],
                        'fy': cam_params['K'][1, 1],
                        'cx': cam_params['K'][0, 2],
                        'cy': cam_params['K'][1, 2],
                        'k': cam_params['distCoef'][[0, 1, 4]].reshape(3, 1),
                        'p': cam_params['distCoef'][[2, 3]].reshape(2, 1)
                    }

                    db[img_idx]['images'].append(image_file)
                    db[img_idx]['cameras'].append(updated_cam_params)
        return db

    def _get_cam(self, seq):
        """
        Loads camera calibration files and returns a dictionary with parameters.
        :param seq: The sequence (folder) name.
        :return: Dictionary mapping (panel, node) to camera parameters.
        """
        cam_file = osp.join(self.dataset_root, seq, f'calibration_{seq}.json')
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                cameras[(cam['panel'], cam['node'])] = {
                    'K': np.array(cam['K']),
                    'distCoef': np.array(cam['distCoef']),
                    'R': np.array(cam['R']).dot(M),
                    't': np.array(cam['t']).reshape((3, 1))
                }
        return cameras

    def __getitem__(self, idx):
        """
        Retrieves multiple images corresponding to the same frame and their camera metadata.
        :param idx: Index of the sample.
        :return: List of images, metadata for each camera.
        """
        db_rec = self.db[idx]
        images = []
        meta = []

        for img_file, cam_params in zip(db_rec['images'], db_rec['cameras']):
            data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if data_numpy is None:
                raise ValueError(f'Failed to load {img_file}')

            height, width, _ = data_numpy.shape
            center = np.array([width / 2.0, height / 2.0])
            scale = get_scale((width, height), self.image_size)
            rotation = 0
            trans = get_affine_transform(center, scale, rotation, self.image_size)
            input_img = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])),
                                       flags=cv2.INTER_LINEAR)

            if self.transform:
                input_img = self.transform(input_img)

            images.append(input_img)
            meta.append({
                'image': img_file,
                'center': center,
                'scale': scale,
                'rotation': rotation,
                'camera': cam_params
            })

        return images, meta

    def __len__(self):
        return self.db_size
