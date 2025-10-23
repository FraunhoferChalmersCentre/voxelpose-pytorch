import numpy as np
import pandas as pd
import cv2
import torch
from lib.utils.definitions import M


def get_heatmap_center(heatmap, threshold=0.5):
    """
    Extract the center point from a heatmap.
    
    Args:
        heatmap: 2D numpy array representing the heatmap
        threshold: minimum confidence threshold
    
    Returns:
        np.array: [x, y] coordinates or [nan, nan] if below threshold
    """
    if heatmap.max()<0.5:
        return np.array([np.nan,np.nan])
    # Find maximum value coordinates
    max_val = heatmap.max()
    max_locs = np.argwhere(heatmap == max_val)
    center = max_locs.mean(axis=0)  # [y, x] as floats

    return center[::-1]


def extract_2d_joints_from_heatmaps(all_heatmaps, metas, threshold=0.5):
    """
    Extract 2D joint locations from heatmaps for all views.
    
    Args:
        heatmaps: List of heatmap tensors, one per view [batch, joints, height, width]
        metas: List of metadata dicts, one per view
        threshold: minimum confidence threshold for joint detection
    
    Returns:
        list: Joint data dicts with keys ['x', 'y', 'joint_idx', 'view_idx', 'image']
    """

    batch_size = all_heatmaps[0][0].shape[0]
    num_joints = all_heatmaps[0][0].shape[1]
    scene_id = 0 
    joints_data = []

    # Loop through superbatches (n_views containing batches of heatmaps)
    for superbatch_idx, superbatch in enumerate(all_heatmaps):
        # Loop through images (scenes) in this superbatch
        for img_idx in range(batch_size):
            scene_id = superbatch_idx * batch_size + img_idx
            # Process all views for this scene
            for view_idx, heatmaps in enumerate(superbatch):
                original_img_paths = metas[superbatch_idx][view_idx]['image']
                original_img = cv2.imread(original_img_paths[img_idx])
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                # Process all joints for this view
                for joint_idx in range(num_joints):
                    heatmap = heatmaps[img_idx, joint_idx, :, :]
                    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

                    # Get heatmap center
                    joint_pts = get_heatmap_center(heatmap)

                    joints_data.append({'x':joint_pts[0],
                                        'y':joint_pts[1],
                                        'joint_idx':joint_idx,
                                        'view_idx':view_idx,
                                        'scene':scene_id,
                                        'img_path':original_img_paths[img_idx]})
    return pd.DataFrame(joints_data)


def triangulate_joints_from_2d(joint_df, metas, scene_idx=0):
    """
    Triangulate 3D joint locations from multi-view 2D detections.
    
    Args:
        joint_data: List of dicts with keys ['x', 'y', 'joint_idx', 'view_idx', 'image']
        metas: List of metadata dicts, one per view containing camera parameters
        img_idx: Which image slice to take intrinsics/extrinsics from
    
    Returns:
        np.ndarray: (num_joints, 3) array of 3D points in classical Z-up coords
                   Missing/untriangulated joints are NaN
    """
    # Convert to DataFrame for easier manipulation
    
    # Since we're processing one image at a time, we can simplify this
    df_pivot = joint_df.pivot_table(index='joint_idx',
                              columns='view_idx', 
                              values=['x', 'y'])
    
    num_joints = joint_df['joint_idx'].nunique()
    view_indices = joint_df['view_idx'].unique()
    ref_view_idx = view_indices.min()
    
    # Pre-compute projection matrices for every camera
    proj_mats = {}
    R_ref_classic = np.array(metas[0][ref_view_idx]['camera']['R'][0])
    t_ref_classic = np.array(metas[0][ref_view_idx]['camera']['T'][0]).reshape(3, 1)

    R_ref_cv = R_ref_classic @ M.T
    t_ref_cv = -M.T @ R_ref_classic.T @ t_ref_classic
    
    for v_idx in view_indices:
        # Intrinsics
        K = np.array(metas[0][v_idx]['camera']['K'][0])

        # Extrinsics in classic camera coords
        R_classic = np.array(metas[0][v_idx]['camera']['R'][0])
        t_classic = np.array(metas[0][v_idx]['camera']['T'][0]).reshape(3, 1)

        # Convert to OpenCV axis convention
        R_cv = R_classic @ M.T
        t_cv = -M.T @ R_classic.T @ t_classic
        
        # Relative pose w.r.t. reference camera in OpenCV coords
        if v_idx == ref_view_idx:
            R_rel = np.eye(3)
            t_rel = np.zeros((3, 1))
        else:
            R_rel = R_cv @ R_ref_cv.T
            t_rel = t_cv - R_rel @ t_ref_cv
        
        # Final 3×4 projection matrix
        proj_mats[v_idx] = K @ np.hstack((R_rel, t_rel))
    
    # Initialize results for just this image
    results = np.full((num_joints, 3), np.nan)
    
    # Triangulate joint-by-joint
    for j_idx, row in df_pivot.iterrows():
        # Collect all valid (x,y,P) for this joint
        pts2d, Ps = [], []
        for v_idx in view_indices:
            if not pd.isna(row.get(('x', v_idx), np.nan)):
                x = row[('x', v_idx)]
                y = row[('y', v_idx)]
                pts2d.append([x, y])
                Ps.append(proj_mats[v_idx])
        
        m = len(Ps)  # how many views for this joint?
        if m < 2:
            continue  # < 2 views → cannot triangulate
        
        pts2d = np.asarray(pts2d, dtype=np.float64)
        
        # Exactly two views
        if m == 2:
            P1, P2 = Ps
            pt1 = pts2d[0,:].reshape(2, 1)
            pt2 = pts2d[1,:].reshape(2, 1)
            
            X_h = cv2.triangulatePoints(P1, P2, pt1, pt2)
            X = (X_h[:3] / X_h[3]).ravel()
        
        # Three or more views
        else:
            A_rows = []
            for (x, y), P in zip(pts2d, Ps):
                A_rows.append(x * P[2] - P[0])
                A_rows.append(y * P[2] - P[1])
            A = np.vstack(A_rows)
            _, _, Vt = np.linalg.svd(A)
            X_h = Vt[-1]
            X = X_h[:3] / X_h[3]
        
        # Convert to classical Z-up coordinates
        results[j_idx] = X @ M
    
    return results


def triangulate_from_heatmaps(heatmaps, metas, threshold=0.5):
    """
    Complete pipeline: extract 2D joints from heatmaps and triangulate to 3D.
    
    Args:
        heatmaps: List of heatmap tensors, one per view [batch, joints, height, width]
        metas: List of metadata dicts, one per view containing camera parameters
        threshold: minimum confidence threshold for joint detection
    
    Returns:
        np.ndarray: (batch_size, num_joints, 3) array of 3D joint coordinates
    """
    # Extract 2D joint locations from heatmaps
    joint_data = extract_2d_joints_from_heatmaps(heatmaps, metas, threshold)
    
    all_results = []

    for scene_idx in joint_data['scene'].unique():
        # Filter joint data for this specific image
        scene_joint_data = joint_data[joint_data['scene'] == scene_idx]
        
        # Triangulate for this specific image using the correct camera parameters
        joints_3d = triangulate_joints_from_2d(scene_joint_data, metas, scene_idx=scene_idx)
        all_results.append(joints_3d)
    
    # Stack results: (batch_size, num_joints, 3)
    return np.stack(all_results, axis=0)