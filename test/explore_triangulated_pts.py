# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
filename = r"C:\Users\ViktorSkantze\OneDrive - Fraunhofer-Chalmers Centre\Python\projects\Pose estimation\results\inference\all_joints.xlsx"

df = pd.read_excel(filename)
# %%
JOINTS_DEF = {
    0: 'neck', 1: 'nose', 2: 'mid-hip', 3: 'l-shoulder', 4: 'l-elbow',
    5: 'l-wrist', 6: 'l-hip', 7: 'l-knee', 8: 'l-ankle', 9: 'r-shoulder',
    10: 'r-elbow', 11: 'r-wrist', 12: 'r-hip', 13: 'r-knee', 14: 'r-ankle'
}

SKELETON_BONES = [
    (0, 1),      # neck to nose
    (0, 3), (3, 4), (4, 5),      # neck -> l-shoulder -> l-elbow -> l-wrist
    (0, 9), (9, 10), (10, 11),   # neck -> r-shoulder -> r-elbow -> r-wrist
    (0, 2),                      # neck to mid-hip
    (2, 6), (6, 7), (7, 8),      # mid-hip -> l-hip -> l-knee -> l-ankle
    (2, 12), (12, 13), (13, 14)  # mid-hip -> r-hip -> r-knee -> r-ankle
]

# --- Prepare the data for all images ---
images = []
image_indices = []

for image_idx, image_df in df.groupby('image'):
    view0_df = image_df[image_df['view_idx'] == 0]
    try:
        x = view0_df['X_3d'].values.astype(float)
        y = view0_df['Y_3d'].values.astype(float)
        z = view0_df['Z_3d'].values.astype(float)
    except KeyError:
        continue

    # Rotate
    x_new = x
    y_new = -z
    z_new = y

    valid_mask = ~(np.isnan(x_new) | np.isnan(y_new) | np.isnan(z_new))
    if np.sum(valid_mask) == 0:
        continue

    images.append((x_new, y_new, z_new, valid_mask))
    image_indices.append(image_idx)

if not images:
    raise ValueError("No valid images found.")

# --- Plot with slider ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initial plot
x, y, z, valid_mask = images[0]
sc = ax.scatter(x[valid_mask], y[valid_mask], z[valid_mask], s=50, c='b', depthshade=False)
lines = []
for joint_start, joint_end in SKELETON_BONES:
    if joint_start < len(x) and joint_end < len(x):
        if valid_mask[joint_start] and valid_mask[joint_end]:
            l, = ax.plot([x[joint_start], x[joint_end]],
                         [y[joint_start], y[joint_end]],
                         [z[joint_start], z[joint_end]], c='k', linewidth=2)
            lines.append(l)
ax.set(xlabel='X', ylabel='Y', zlabel='Z', title=f'3D Joints: Image {image_indices[0]}')
ax.view_init(elev=0, azim=-90)

# Slider setup
ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05])
slider = Slider(ax_slider, 'Image', 0, len(images)-1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    x, y, z, valid_mask = images[idx]
    # Update scatter
    sc._offsets3d = (x[valid_mask], y[valid_mask], z[valid_mask])
    # Remove old lines
    for l in lines:
        l.remove()
    lines.clear()
    # Draw new skeleton lines
    for joint_start, joint_end in SKELETON_BONES:
        if joint_start < len(x) and joint_end < len(x):
            if valid_mask[joint_start] and valid_mask[joint_end]:
                l, = ax.plot([x[joint_start], x[joint_end]],
                             [y[joint_start], y[joint_end]],
                             [z[joint_start], z[joint_end]], c='k', linewidth=2)
                lines.append(l)
    ax.set_title(f'3D Joints: Image {image_indices[idx]}')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()