# Import the libraries we need
from matplotlib.pyplot import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dtcwt.compat import dtwavexfm3, dtwaveifm3
from dtcwt.coeffs import biort, qshift
import dtcwt
# Specify details about sphere and grid size
def main():
    GRID_SIZE = 128
    SPHERE_RAD = int(0.45 * GRID_SIZE) + 0.5

    # Compute an image of the sphere
    grid = np.arange(-(GRID_SIZE >> 1), GRID_SIZE >> 1)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    sphere = (0.5 + np.clip(SPHERE_RAD - r, -0.5, 0.5)).astype(np.float32)

    # Specify number of levels and wavelet family to use
    nlevels = 1
    b = biort('near_sym_b_bp')  # 可以选择 'near_sym_a', 'near_sym_b', 'legall', 'antonini'
    q = qshift('qshift_a')  # 可以选择 'qshift_a', 'qshift_b', 'qshift_c', 'qshift_d'

    # Perform the 3D DT-CWT of the sphere
    # Yl, Yh = dtwavexfm3(sphere, nlevels, b, q, discard_level_1=False)
    transform = dtcwt.Transform3d(b,q)
    mandrill_t = transform.forward(sphere,nlevels=nlevels) 
    Yl=mandrill_t.lowpass
    Yh=mandrill_t.highpasses

    # Tolerance for zero detection
    tolerance = 0.2

    # Plot the sphere and mark directions
    figure(figsize=(10, 10))
    ax = gcf().add_subplot(1, 1, 1, projection='3d')
    ax.set_aspect('equal')
    ax.view_init(35, 45)

    # Plot the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='w', edgecolor=(0.6, 0.6, 0.6), alpha=0.3)

    scale = 1.1

    for idx in range(Yh[-1].shape[3]):
        Z = Yh[-1][:, :, :, idx]
        C = np.abs(Z)
        max_loc = np.asarray(np.unravel_index(np.argmax(C), C.shape)) - np.asarray(C.shape) * 0.5
        max_loc /= np.sqrt(np.sum(max_loc * max_loc))  # Normalize

        # Apply tolerance to treat values close to 0 as 0
        max_loc = np.where(np.abs(max_loc) < tolerance, 0, max_loc)
        
        # Print the index and direction for verification
        print(f"Subband {idx + 1}: Direction {max_loc}")
        
        # Determine the direction label for each axis based on max_loc values
        x_label = "+x" if max_loc[0] > 0 else ("-x" if max_loc[0] < 0 else "0")
        y_label = "+y" if max_loc[1] > 0 else ("-y" if max_loc[1] < 0 else "0")
        z_label = "+z" if max_loc[2] > 0 else ("-z" if max_loc[2] < 0 else "0")

        # Identify type of direction: axis-aligned or diagonal
        non_zero_count = np.count_nonzero(max_loc)  # Count non-zero components
        
        if non_zero_count == 1:
            # Axis-aligned direction (one non-zero component)
            font_color = 'blue'
            direction_type = "Axis-Aligned"
        else:
            # Diagonal direction (two or three non-zero components)
            font_color = 'green'
            direction_type = "Diagonal"

        # Construct the direction label with the type for clarity
        direction_label = f"Subband {idx + 1} ({x_label}, {y_label}, {z_label}) - {direction_type}"
        
        # Apply a larger random offset to avoid overlapping
        offset = (np.random.rand(3) - 0.5) * 0.2  # Larger offset range
        text_position = max_loc * scale + offset
        
        ax.text(text_position[0], text_position[1], text_position[2], direction_label, 
                fontsize=6, color=font_color)  # Reduced font size

    # Scatter plot to mark all directions
    locs = np.array([max_loc])
    ax.scatter(locs[:, 0] * scale, locs[:, 1] * scale, locs[:, 2] * scale, c=np.arange(locs.shape[0]))

    w = 1.2
    ax.set_xlim([-w, w])
    ax.set_ylim([-w, w])
    ax.set_zlim([-w, w])

    title('3D DT-CWT subband directions on entire sphere (Axis-Aligned in Blue, Diagonal in Green)')
    tight_layout()

    show()

def sphere_to_xyz(r, theta, phi):
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return r * np.asarray((st * cp, st * sp, ct))

if __name__ == '__main__':
    main()
