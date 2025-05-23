import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Joint definition
JOINTS_DEF = {
    0: 'neck', 1: 'nose', 2: 'mid-hip', 3: 'l-shoulder', 4: 'l-elbow',
    5: 'l-wrist', 6: 'l-hip', 7: 'l-knee', 8: 'l-ankle', 9: 'r-shoulder',
    10: 'r-elbow', 11: 'r-wrist', 12: 'r-hip', 13: 'r-knee', 14: 'r-ankle'
}

def visualize_heatmaps(views, all_heatmaps, output_dir, meta=None, mean=None, std=None, idx=0, heatmapsize=False):
    """
    Visualizes heatmaps overlayed on original images and saves them.

    Args:
        views (list[torch.Tensor]): List of input images from different views.
        all_heatmaps (list[np.array]): Corresponding heatmaps for each view.
        output_dir (str): Path where the visualizations will be saved.
        meta (list[dict], optional): Metadata containing original image paths.
        mean (list, optional): Mean values for denormalization.
        std (list, optional): Std values for denormalization.
        idx (int, optional): Index of the image in the batch to visualize. Default is 0.
        heatmapsize (bool, optional): If True, scales the image down to match heatmap size.
    """
    os.makedirs(output_dir, exist_ok=True)

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    num_views = len(views)
    num_joints = all_heatmaps[0].shape[1]

    for view_idx in range(num_views):
        view_folder = os.path.join(output_dir, f"view_{view_idx}")
        os.makedirs(view_folder, exist_ok=True)

        if meta is not None:
            original_img_path = meta[view_idx]['image'][idx]
            original_img = cv2.imread(original_img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        else:
            image = views[view_idx][idx].cpu().numpy().transpose(1, 2, 0)
            image = image * np.array(std) + np.array(mean)
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            original_img = image

        for joint_idx in range(num_joints):
            heatmap = all_heatmaps[view_idx][idx, joint_idx, :, :].clone().cpu().numpy()

            if heatmapsize:
                # Scale image to heatmap size
                resized_img = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))
                overlayed_img = overlay_heatmap(resized_img, heatmap)
            else:
                # Scale heatmap to image size
                heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                overlayed_img = overlay_heatmap(original_img, heatmap)

            # Give name to joints
            joint_name = JOINTS_DEF.get(joint_idx, f"joint_{joint_idx}")

            output_file = os.path.join(view_folder, f"{joint_name}.jpg")
            plt.imsave(output_file, overlayed_img)

        print(f"✅ Saved heatmaps for view {view_idx}")

    print(f"✅ Heatmap visualization completed. Results saved to {output_dir}")


def overlay_heatmap(image, heatmap):
    """
    Overlay a heatmap onto an image.
    :param image: Original image (numpy array).
    :param heatmap: Heatmap (PyTorch tensor or numpy array).
    :return: Image with heatmap overlay.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()

    heatmap = np.uint8(255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))  # Normalize and convert to uint8
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Convert grayscale heatmap to color
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)  # Blend images
    return overlay


def save_interactive_3d_heatmaps_world(heatmap_3d, space_size, space_center,
                                       output_dir, meta=None):
    """
    Saves a 3D scatter plot for each joint's heatmap in world coordinates.
    If meta is provided, camera positions and viewing directions are also plotted.
    Each joint is saved as an HTML file in the output directory.

    Args:
        heatmap_3d (torch.Tensor or numpy.ndarray): 3D heatmap tensor with shape
            (num_joints, W, D, H).
        space_size (tuple): Size of the 3D space in real-world units (X, Y, Z) in mm.
        space_center (tuple): Center of the 3D space in world coordinates (X, Y, Z).
        output_dir (str): Directory where HTML files will be saved.
        meta (list[dict], optional): List of metadata dictionaries containing camera parameters.
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # If heatmap_3d is a torch tensor, convert to numpy
    if isinstance(heatmap_3d, torch.Tensor):
        heatmap_3d = heatmap_3d.clone().cpu().numpy()

    num_joints = heatmap_3d.shape[0]

    # Loop over joints and create a plot for each
    for joint_idx in range(num_joints):
        joint_name = JOINTS_DEF.get(joint_idx, f'joint{joint_idx}')
        output_path = os.path.join(output_dir, f"scatter_3d_heatmap_{joint_name}.html")

        # Get the heatmap for this joint; shape: (W, D, H)
        joint_heatmap = heatmap_3d[joint_idx]
        W, D, H = joint_heatmap.shape

        # Convert voxel indices to world coordinates
        x_idx, y_idx, z_idx = np.meshgrid(
            np.arange(W), np.arange(D), np.arange(H), indexing='ij'
        )
        x_world = x_idx / (W - 1) * space_size[0] + space_center[0] - space_size[0] / 2
        y_world = y_idx / (D - 1) * space_size[1] + space_center[1] - space_size[1] / 2
        z_world = z_idx / (H - 1) * space_size[2] + space_center[2] - space_size[2] / 2

        # Flatten the arrays
        x = x_world.flatten()
        y = y_world.flatten()
        z = z_world.flatten()
        intensity = joint_heatmap.flatten().astype(np.float32)

        # Create a mask based on an initial threshold
        base_threshold = 0.2
        mask = intensity > base_threshold * intensity.max()
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        intensity_filtered = intensity[mask]

        # Create the figure
        fig = go.Figure()

        # Add the heatmap trace
        heatmap_trace = go.Scatter3d(
            x=x_filtered, y=y_filtered, z=z_filtered,
            mode='markers',
            marker=dict(size=3, color=intensity_filtered, colorscale='Jet', opacity=0.7),
            name=f"3D Heatmap: {joint_name}"
        )
        fig.add_trace(heatmap_trace)

        # Process camera metadata if provided, and collect camera positions for axis adjustment
        all_cam_positions = []
        if meta:
            camera_traces = []
            for i, cam in enumerate(meta):
                cam_pos = np.array(cam['camera']['T'][0].cpu().numpy()).flatten()  # (3,)
                cam_rot = np.array(cam['camera']['R'][0].cpu().numpy())  # (3,3)
                all_cam_positions.append(cam_pos)

                # Scale the viewing direction based on the distance from space center
                direction_scale = np.linalg.norm(cam_pos - np.array(space_center)) / 10.0

                # Calculate camera direction using the transposed rotation matrix.
                # This converts the rotation from world-to-camera to camera-to-world.
                cam_dir = cam_pos + cam_rot.T[:, 2] * direction_scale

                # Plot camera position as a marker
                cam_point = go.Scatter3d(
                    x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='circle'),
                    name=f"Camera {i}",
                    showlegend=True
                )
                camera_traces.append(cam_point)

                # Plot camera direction as a line
                cam_line = go.Scatter3d(
                    x=[cam_pos[0], cam_dir[0]],
                    y=[cam_pos[1], cam_dir[1]],
                    z=[cam_pos[2], cam_dir[2]],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=f"Camera {i} Direction",
                    showlegend=False
                )
                camera_traces.append(cam_line)

            fig.add_traces(camera_traces)

        # Create slider steps for threshold adjustment
        steps = []
        for t in np.linspace(0.05, 1.0, 20):
            mask_t = intensity > t * intensity.max()
            step = dict(
                method="restyle",
                args=[{
                    "x": [x[mask_t]],
                    "y": [y[mask_t]],
                    "z": [z[mask_t]],
                    "marker.color": [intensity[mask_t]],
                    "marker.size": [3],
                    "mode": "markers"
                }, [0]],  # Update only the heatmap trace (index 0)
                label=f"{t:.2f}"
            )
            steps.append(step)

        sliders = [dict(
            active=3,
            currentvalue={"prefix": "Threshold: "},
            pad={"t": 50},
            steps=steps
        )]

        # Compute axis limits based on the provided space parameters
        min_x = space_center[0] - space_size[0] / 2
        max_x = space_center[0] + space_size[0] / 2
        min_y = space_center[1] - space_size[1] / 2
        max_y = space_center[1] + space_size[1] / 2
        min_z = space_center[2] - space_size[2] / 2
        max_z = space_center[2] + space_size[2] / 2

        # If meta is provided, update limits based on camera positions
        if meta and all_cam_positions:
            all_cam_positions = np.array(all_cam_positions)
            cam_min = all_cam_positions.min(axis=0)
            cam_max = all_cam_positions.max(axis=0)
            margin = 200  # Margin to ensure cameras are not at the plot boundary

            min_x = min(min_x, cam_min[0] - margin)
            min_y = min(min_y, cam_min[1] - margin)
            min_z = min(min_z, cam_min[2] - margin)
            max_x = max(max_x, cam_max[0] + margin)
            max_y = max(max_y, cam_max[1] + margin)
            max_z = max(max_z, cam_max[2] + margin)

        # Update the layout with slider and axis ranges
        fig.update_layout(
            title=f"3D Heatmap in World Coordinates: {joint_name}",
            scene=dict(
                xaxis=dict(title="X (mm)", range=[min_x, max_x]),
                yaxis=dict(title="Y (mm)", range=[min_y, max_y]),
                zaxis=dict(title="Z (mm)", range=[min_z, max_z])
            ),
            sliders=sliders,
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )

        # Save the figure as an HTML file
        fig.write_html(output_path)
        print(f"✅ Interactive 3D heatmap for '{joint_name}' saved as {output_path}")


def save_interactive_3d_heatmap_world(heatmap_3d, space_size, space_center,
                                      output_path="scatter_3d_heatmap.html",
                                      meta=None):
    """
    Saves a 3D scatter plot of the heatmap in world coordinates.
    If meta is provided, camera positions and viewing directions are also plotted.

    Args:
        heatmap_3d (torch.Tensor or numpy.ndarray): 3D heatmap with shape (W, D, H).
        space_size (tuple): Size of the 3D space in real-world units (X, Y, Z) in mm.
        space_center (tuple): Center of the 3D space in world coordinates (X, Y, Z).
        output_path (str): Path where the interactive HTML file will be saved.
        meta (list[dict], optional): List of metadata dictionaries containing camera parameters.
    """

    # Convert heatmap to numpy array if it is a PyTorch tensor
    if isinstance(heatmap_3d, torch.Tensor):
        heatmap_np = heatmap_3d.clone().cpu().numpy()
    else:
        heatmap_np = heatmap_3d

    W, D, H = heatmap_np.shape

    # Convert voxel indices to world coordinates
    x_idx, y_idx, z_idx = np.meshgrid(
        np.arange(W), np.arange(D), np.arange(H), indexing='ij'
    )

    x_world = x_idx / (W - 1) * space_size[0] + space_center[0] - space_size[0] / 2
    y_world = y_idx / (D - 1) * space_size[1] + space_center[1] - space_size[1] / 2
    z_world = z_idx / (H - 1) * space_size[2] + space_center[2] - space_size[2] / 2

    # Flatten the arrays
    x = x_world.flatten()
    y = y_world.flatten()
    z = z_world.flatten()
    intensity = heatmap_np.flatten().astype(np.float32)

    # Create a mask based on an initial threshold
    base_threshold = 0.2
    mask = intensity > base_threshold * intensity.max()
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    intensity_filtered = intensity[mask]

    # Create the figure
    fig = go.Figure()

    # Add heatmap trace
    heatmap_trace = go.Scatter3d(
        x=x_filtered, y=y_filtered, z=z_filtered,
        mode='markers',
        marker=dict(size=3, color=intensity_filtered, colorscale='Jet', opacity=0.7),
        name="3D Heatmap"
    )
    fig.add_trace(heatmap_trace)

    # Process camera metadata if provided, and collect camera positions for axis adjustment
    all_cam_positions = []
    if meta:
        camera_traces = []
        for i, cam in enumerate(meta):
            cam_pos = np.array(cam['camera']['T'][0].cpu().numpy()).flatten()  # (3,)
            cam_rot = np.array(cam['camera']['R'][0].cpu().numpy())  # (3,3)

            # Save camera position for later axis adjustment
            all_cam_positions.append(cam_pos)

            # Adjust viewing direction scaling based on distance from space center
            direction_scale = np.linalg.norm(cam_pos - np.array(space_center)) / 10.0

            # Compute camera direction using the transposed rotation matrix
            cam_dir = cam_pos + cam_rot.T[:, 2] * direction_scale

            # Plot camera position as a marker
            cam_point = go.Scatter3d(
                x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name=f"Camera {i}",
                showlegend=True
            )
            camera_traces.append(cam_point)

            # Plot camera direction as a line
            cam_line = go.Scatter3d(
                x=[cam_pos[0], cam_dir[0]],
                y=[cam_pos[1], cam_dir[1]],
                z=[cam_pos[2], cam_dir[2]],
                mode='lines',
                line=dict(color='red', width=3),
                name=f"Camera {i} Direction",
                showlegend=False
            )
            camera_traces.append(cam_line)

        fig.add_traces(camera_traces)

    # Create slider steps for threshold adjustment
    steps = []
    for t in np.linspace(0.05, 1.0, 20):
        mask_t = intensity > t * intensity.max()
        step = dict(
            method="restyle",
            args=[{
                "x": [x[mask_t]],
                "y": [y[mask_t]],
                "z": [z[mask_t]],
                "marker.color": [intensity[mask_t]],
                "marker.size": [3],
                "mode": "markers"
            }, [0]],  # Update only the heatmap trace (index 0)
            label=f"{t:.2f}"
        )
        steps.append(step)

    sliders = [dict(
        active=3,
        currentvalue={"prefix": "Threshold: "},
        pad={"t": 50},
        steps=steps
    )]

    # Compute axis limits
    # a) Standard: based on space_center ± space_size/2
    min_x = space_center[0] - space_size[0] / 2
    max_x = space_center[0] + space_size[0] / 2
    min_y = space_center[1] - space_size[1] / 2
    max_y = space_center[1] + space_size[1] / 2
    min_z = space_center[2] - space_size[2] / 2
    max_z = space_center[2] + space_size[2] / 2

    # b) If meta is provided, update limits based on camera positions
    if meta and len(all_cam_positions) > 0:
        all_cam_positions = np.array(all_cam_positions)  # shape = (N, 3)
        cam_min = all_cam_positions.min(axis=0)
        cam_max = all_cam_positions.max(axis=0)

        margin = 200  # Margin to ensure cameras are not at the plot boundary
        min_x = min(min_x, cam_min[0] - margin)
        min_y = min(min_y, cam_min[1] - margin)
        min_z = min(min_z, cam_min[2] - margin)

        max_x = max(max_x, cam_max[0] + margin)
        max_y = max(max_y, cam_max[1] + margin)
        max_z = max(max_z, cam_max[2] + margin)

    # Update layout with slider and axis ranges
    fig.update_layout(
        title="3D Heatmap in World Coordinates",
        scene=dict(
            xaxis=dict(title="X (mm)", range=[min_x, max_x]),
            yaxis=dict(title="Y (mm)", range=[min_y, max_y]),
            zaxis=dict(title="Z (mm)", range=[min_z, max_z])
        ),
        sliders=sliders,
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )

    fig.write_html(output_path)
    print(f"✅ Interactive 3D heatmap with slider saved as {output_path}")


LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

def save_interactive_3d_skeleton_with_slider(pred, limbs=None, output_path="skeleton_3d_slider.html", meta=None):
    """
    Save an interactive 3D skeleton visualization with slider for frames.

    Args:
        pred (torch.Tensor): Shape (B, N, J, 5), J=15, last dim: (x, y, z, present_flag, confidence).
        limbs (list of tuple): List of joint index pairs.
        output_path (str): Output path to HTML file.
        meta (list, optional): List of metadata dicts for visualizing camera positions.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if limbs is None:
        limbs = LIMBS

    fig = go.Figure()
    B, N, J, _ = pred.shape
    frame_list = []

    # Create traces for each frame
    for b in range(B):
        frame_traces = []
        for n in range(N):
            skeleton = pred[b, n]  # (J, 5)
            present = skeleton[:, 3] >= 0
            if not present.any():
                continue

            x, y, z = skeleton[:, 0], skeleton[:, 1], skeleton[:, 2]

            # Joints
            joint_trace = go.Scatter3d(
                x=x[present], y=y[present], z=z[present],
                mode='markers+text',
                text=[str(i) for i in range(J)],
                marker=dict(size=4, color='blue'),
                name=f'Person {n}',
                showlegend=False,
                visible=(b == 0)
            )
            fig.add_trace(joint_trace)
            frame_traces.append(joint_trace)

            # Limbs
            for i, j in limbs:
                if present[i] and present[j]:
                    limb_trace = go.Scatter3d(
                        x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                        mode='lines',
                        line=dict(width=3, color='blue'),
                        showlegend=False,
                        visible=(b == 0)
                    )
                    fig.add_trace(limb_trace)
                    frame_traces.append(limb_trace)

        frame_list.append(frame_traces)

    # Add camera positions and directions if available
    if meta:
        for i, m in enumerate(meta):
            T = m["camera"]["T"][0].cpu().numpy().flatten()
            R = m["camera"]["R"][0].cpu().numpy()
            cam_pos = T
            direction = cam_pos + R.T[:, 2] * 200

            cam_trace = go.Scatter3d(
                x=[cam_pos[0]], y=[cam_pos[1]], z=[cam_pos[2]],
                mode='markers',
                marker=dict(size=6, color='red', symbol='circle'),
                name=f"Camera {i}",
                showlegend=False,
                visible=True
            )
            dir_trace = go.Scatter3d(
                x=[cam_pos[0], direction[0]],
                y=[cam_pos[1], direction[1]],
                z=[cam_pos[2], direction[2]],
                mode='lines',
                line=dict(color='red', width=3),
                showlegend=False,
                visible=True
            )
            fig.add_trace(cam_trace)
            fig.add_trace(dir_trace)

    # Build slider steps
    steps = []
    total_skeleton_traces = sum(len(f) for f in frame_list)
    for i in range(B):
        visibility = [False] * total_skeleton_traces
        offset = 0
        for b in range(i):
            offset += len(frame_list[b])
        for j in range(len(frame_list[i])):
            visibility[offset + j] = True
        # Add camera traces at the end
        visibility += [True] * (len(fig.data) - total_skeleton_traces)

        step = dict(
            method="update",
            args=[{"visible": visibility}],
            label=f"Frame {i}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frame: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        title="3D Skeletons Over Time",
        sliders=sliders,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(output_path)
    print(f"\u2705 Skeleton slider visualization saved as {output_path}")

