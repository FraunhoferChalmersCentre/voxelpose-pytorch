# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # or 'QtAgg' depending on your installation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
filename = r"C:\Users\ViktorSkantze\OneDrive - Fraunhofer-Chalmers Centre\Python\projects\Pose estimation\results\inference\all_joints.xlsx"

df = pd.read_excel(filename)
# %%

for image_idx, image_df in df.groupby('image'):
    view0_df = image_df[image_df['view_idx'] == 0]
    if image_idx == 3:
        break
    # Convert to numpy arrays with validation
    try:
        x = view0_df['X_3d'].values.astype(float)
        y = view0_df['Y_3d'].values.astype(float)
        z = view0_df['Z_3d'].values.astype(float)
    except KeyError:
        print(f"Missing columns in image {image_idx}")
        continue

    # Rotate
    x_new = y
    y_new = -z
    z_new = x

    # Comprehensive data check
    valid_mask = ~(np.isnan(x_new) | np.isnan(y_new) | np.isnan(z_new))
    if np.sum(valid_mask) == 0:
        print(f"Skipping image {image_idx} - no valid 3D points")
        continue

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot only valid points
    ax.scatter(x_new[valid_mask], y_new[valid_mask], z_new[valid_mask],
               s=50, c='b', depthshade=False)

    ax.set(xlabel='X', ylabel='Y', zlabel='Z',
           title=f'3D Joints: Image {image_idx}')
    ax.view_init(elev=0, azim=-90)
    plt.show()