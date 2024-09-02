import camera_convert
import core
import matplotlib.pyplot as plt
import numpy as np
from camera_convert import CameraState

DIFF_LEN = 0.1
IDEN_TIMES = 1000
DATA_NUM = 7
MINIMUM_ERROR = 35

MAX_CHANGE = 1
CHANGING_RANGE = [5, 100, 100, 20, 1, 1, 10, 10]
ORIGIN_VALUE = [90, 18, -442, 55.9, 0.9, 0, 51.45, 51.09]

ENABLE_SMOOTH_FACTOR = False
OVERLAY_DISTANCE = 40  # The distance criterion of endpoints, deciding whether to merge two walls
LAMBDA = 0


def partial_dirivative(
    camera_xyz: tuple[float, float, float],
    camera_rotation: tuple[float, float, float],
    fov: tuple[float, float],
    dt: str,
    cords: list,
) -> np.ndarray:
    """
    Calculate the partial derivative of the camera parameters with respect to the given change in dt.

    Args:
        image (cv2.UMat): A cv2.UMat object representing the image.
        camera_xyz (tuple): A tuple representing the camera's x, y, and z coordinates.
        camera_rotation (tuple): A tuple representing the camera's rotation angles (theta, phi, omega).
        fov (tuple): A tuple representing the camera's field of view angles (half_fov_h, half_fov_v).
        dt (string): A string representing the parameter to calculate the partial derivative for.

    Returns:
        partial_dirivative (np.ndarray):
            The partial derivative of the camera parameters with respect to the given change in dt.

    Notes:
        The function uses the DIFF_LEN constant for the change in dt.
    """
    cam_0 = CameraState(camera_xyz, camera_rotation, fov, (320, 240))
    cam_1 = CameraState(camera_xyz, camera_rotation, fov, (320, 240))

    if dt == "x":
        cam_1.x += DIFF_LEN
    elif dt == "y":
        cam_1.y += DIFF_LEN
    elif dt == "z":
        cam_1.z += DIFF_LEN

    elif dt == "theta":
        cam_1.theta += DIFF_LEN
    elif dt == "phi":
        cam_1.phi += DIFF_LEN
    elif dt == "omega":
        cam_1.omega += DIFF_LEN

    elif dt == "half_fov_h":
        cam_1.half_fov_h += DIFF_LEN
    elif dt == "half_fov_v":
        cam_1.half_fov_v += DIFF_LEN
    else:
        print("ERROR")

    cam_1.update()

    d_cords = np.array([])  # [dx1, dy1, dz1, dx2, dy2, dz2, ... ]
    for i in range(DATA_NUM):
        _, x1, y1 = camera_convert.img2space(cam_0, cords[i][0], cords[i][1])
        _, x2, y2 = camera_convert.img2space(cam_1, cords[i][0], cords[i][1])
        v1 = (x1, y1)
        v2 = (x2, y2)

        d_cords = np.concatenate((d_cords, np.array(core.vec_sub(v2, v1))))

    return d_cords / DIFF_LEN


def Jacobian(
    camera_xyz: tuple[float, float, float],
    camera_rotation: tuple[float, float, float],
    fov: tuple[float, float],
    cords: np.ndarray,
) -> np.matrix:
    """
    Calculate the Jacobian matrix for a given image with camera position, camera rotation, and field of view.

    Args:
        image (cv2.UMat): The image to be calculated.
        camera_xyz (tuple): The camera position in 3D space (x, y, z).
        camera_rotation (tuple): The camera rotation angles (theta, phi, omega).
        fov (tuple): The field of view angles (half_fov_h, half_fov_v).

    Returns:
        Jacobian (np.matrix): The Jacobian matrix.

    """
    Jacobian = np.asmatrix(
        [
            partial_dirivative(camera_xyz, camera_rotation, fov, "x", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "y", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "z", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "theta", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "phi", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "omega", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "half_fov_h", cords),
            partial_dirivative(camera_xyz, camera_rotation, fov, "half_fov_v", cords),
        ]
    )
    return Jacobian.T


if __name__ == "__main__":

    camera_xyz_0 = np.array([90, 18, -442])
    camera_rotation_0 = np.array([55.9, 0.9, 0])
    fov_0 = np.array([51.45, 51.09])
    resolution = (320, 240)

    cords = [
        (54, 5),
        (88, 22),
        (158, 13),
        (158, 22),
        (158, 37),
        (221, 22),
        (257, 5),
    ]  # coords of stuff in pics
    E_test = np.array(
        [1600, -400, 1200, -200, 1400, 00, 1200, 0, 1000, 0, 1200, 200, 1600, 400]
    )  # coords of stuff in space

    p = np.concatenate((camera_xyz_0, camera_rotation_0, fov_0))  # array of paras
    d_p = np.zeros(8)

    dE_list = []

    for i in range(IDEN_TIMES):
        # Generate camera state objects
        print(f"p = {p}")
        cam = CameraState(tuple(p[0:3]), tuple(p[3:6]), tuple(p[6:8]), resolution)

        # Calculate the ideal position with the current parameters
        E_cal = np.array([])
        for j in range(DATA_NUM):
            _, x, y = camera_convert.img2space(cam, cords[j][0], cords[j][1])
            E_cal = np.concatenate((E_cal, np.array([x, y])))

        E_cal = E_cal.reshape(-1)
        print(f"calculated E = {E_cal}")

        # Calculate the current error
        d_E = (E_test - E_cal).T
        print(f"|dE| = {np.linalg.norm(d_E)}")
        print(f"dE = {d_E}")

        # if np.linalg.norm(d_E) > 3000:
        #     dE_list.append(3000)
        # else:
        dE_list.append(np.linalg.norm(d_E))

        if np.linalg.norm(d_E) < MINIMUM_ERROR:
            break

        # Generate Jacobian matrix (NUM_OF_DATAx16x8)
        J = Jacobian(tuple(p[0:3]), tuple(p[3:6]), tuple(p[6:8]), cords)
        print(f"Jacobian = {J}")
        print()
        # Calculate the parameter compensation with the Jacobian matrix and error
        if ENABLE_SMOOTH_FACTOR:
            # Introduce a smooth factor to make the result curve smoother
            J_Tik = np.linalg.inv(J.T @ J + LAMBDA * np.eye(8)) @ J.T
            d_p = J_Tik @ d_E
        else:
            d_p = np.linalg.pinv(J) @ d_E

        # to avoid p becoming too large
        d_p_r = np.array([np.atan(d_p[0, j]) for j in range(8)])
        d_p_r = MAX_CHANGE * d_p_r / (np.pi / 2)

        # compensate the paras
        p = np.reshape(p, (1, 8))
        p += d_p_r
        p = np.reshape(p, (8,))

        # restrict the range of paras
        # for j in range(len(p)):
        #     p[j] = CHANGING_RANGE[j] * math.atan(p[j] - ORIGIN_VALUE[j]) / (math.pi/2)  + ORIGIN_VALUE[j]

        print(f"p = {p}")
        print(f"d_p = {d_p_r}")
        print(f"loop {i} finished\n")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(dE_list)
    plt.show()
