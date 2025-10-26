from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Chessboard size constants
SQUARE_SIZE = 0.0205
NUM_CORNERS_X = 9
NUM_CORNERS_Y = 7

def generate_chessboard_3D_world_coordinates() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 3D points of chessboard cross-junctions in the world coordinate frame. The order
    of the points in the output arrays should be left-to-right, top-to-bottom.

    Returns:
        X_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) Numpy array of world points along the x-axis.
        Y_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) Numpy array of world points along the y-axis.
        Z_world: (NUM_CORNERS_X * NUM_CORNERS_Y,) Numpy array of world points along the z-axis.
    """
    ########## Code starts here ##########
    # Hint: Use the SQUARE_SIZE, NUM_CORNERS_X, and NUM_CORNERS_Y
    # Hint: Use np.meshgrid to generate the X, Y coordinates

    ########## Code ends here ##########

    # Sanity check the order is correct
    if (X_world[0] > X_world[-1] or Y_world[0] < Y_world[-1]):
        raise ValueError("Check that order of the coordinates is correct")
    
    return X_world, Y_world, Z_world

def generate_chessboard_2D_pixel_coordinates(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D pixel coordinates chessboard cross-junctions. The order of the points
    in the output arrays should be left-to-right, top-to-bottom.

    Args:
        img: Path to image.

    Returns:
        u_meas: (NUM_CORNERS_X * NUM_CORNERS_Y,) Numpy array of pixel coordinates along the x-axis.
        v_meas: (NUM_CORNERS_X * NUM_CORNERS_Y,) Numpy array of pixel coordinates along the y-axis.
    """
    img = cv2.imread(img_path, 0)

    ########## Code starts here ##########
    # NOTE: Ensure that points are ordered left-to-right, bottom-to-top

    ########## Code ends here ##########
    
    return u_meas, v_meas

def compute_homography(
    u: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Computes homography matrix, which is required to transform 3D world 
    coordinates to 2D pixel coordinates.

    Args:
        u: (N,) pixel x-coordinates.
        v: (N,) pixel y-coordinates.
        X: (N,) 3D world x-coordinates.
        Y: (N,) 3D world y-coordinates.
    
    Returns:
        M: (3, 3) Homography matrix.
    """
    ########## Code starts here ##########
    # Hint: use np.linalg.svd
    # Hint: Use <array>.reshape(3, 3) to give M the proper dimensions.

    return M
    ########## Code ends here ##########
    
def compute_intrinsics(homographies: List[np.ndarray]) -> np.ndarray:
    """
    Computes camera intrinsics from a list of homographies.
    
    Args:
        homographies: List of (3, 3) homography matrices, one per calibration image.
    
    Returns:
        K: (3, 3) camera intrinsic matrix.
    """
    def v(M, i, j):
        return np.array(
            [
                M[0, i] * M[0, j],
                M[0, i] * M[1, j] + M[1, i] * M[0, j],
                M[1, i] * M[1, j],
                M[2, i] * M[0, j] + M[0, i] * M[2, j],
                M[2, i] * M[1, j] + M[1, i] * M[2, j],
                M[2, i] * M[2, j],
            ]
        )

    V = np.array(
        [v(M, 0, 1) for M in homographies] \
            + [v(M, 0, 0) - v(M, 1, 1) for M in homographies]
    )

    ########## Code starts here ##########

    ########## Code ends here ##########
    
def compute_extrinsics(K: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes camera extrinsics from a homography and intrinsics.

    Args:
        K: (3, 3) camera intrinsic matrix.
        H: (3, 3) homography matrix.
    
    Returns:
        R: (3, 3) camera extrinsic rotation matrix.
        t: (3, 1) camera extrinsic translation vector.
    """
    ########## Code starts here ##########
    # Hint: use np.linalg.inv() to compute KinvM = inv(K) @ M
    # Hint: use np.linalg.norm() to compute the normalization constant for r1, r2, and t
    # Hint: use np.cross() to compute r3
    # Hint: use np.column_stack() to combine r1, r2, and r3 into the matrix Q

    ########## Code ends here ##########

    return R, t

def transform_world_to_pixel(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    Transforms 3D point in world coordinates to homogenous point in pixel coordinates.

    Note: Recall the convention, where R, t are the rotation and translation
          of the world frame with respect to the camera frame (not vice versa!)

    Args:
        K: (3, 3) camera intrinsic matrix.
        R: (3, 3) camera extrinsic rotation matrix.
        t: (3, 1) camera extrinsic translation vector.
        P: (3,) 3D point in world coordinates.

    Returns:
        p: (3,) homogenous point in pixel coordinates.
    """
    ########## Code starts here ##########

    ########## Code ends here ##########