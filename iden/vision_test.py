import cv2
import numpy as np

try:
    import camera_convert
    import find_color

except ModuleNotFoundError:
    import os
    import sys

    current_dir = os.path.dirname(__file__)
    sys.path.append(f"{current_dir}")
    import camera_convert
    import find_color
else:
    ...

ENABLE_WEIRD_COLOR_DETECTING = False

CAMERA_STATE = camera_convert.CameraState(
    # (309, 0, -218), (52.8, 2.1, 0.4), (62.2, 62), (640, 480)
    # (295, 12, -221), (57.7, 1.1, 0.3), (51.45, 51.09), (640, 480)
    # (295, 12, -221), (57.7, 1.1, 0.3), (51.45, 51.09), (320, 240)
    # (84, 4, -243), (59.4, 1.1, -0.1), (51.45, 51.09), (640, 480)
    # (78, 13, -164), (69.4, 1., 0.5), (47.66, 33.90), (640, 480)
    # (891.39, -119.73, -112.10),(53.2, 0.9, 0.9),(62.2, 62),(320, 240),
    # (1, -1, -11),(53.2, 0.9, 0.9),(62.2, 62),(320, 240)
    # (-508.5,-1724.0,33.9),(52.1,-13.9,9.8),(115.2,78.6),(320,240)
    # (89.6,117.3,-541.3),(36.7,0.9, 0), (60.9, 59.6), (320,240)
    # (142, -38, -219), (56.3, 0.9, -0.5), (51.45, 51.09), (320, 240)
    # (143.69091796875, -53.24489974975586, -242.48875427246094), (92.91905975341797, 51.92171859741211, -37.98847198486328), (84.68187713623047, 80.89322662353516), (320, 240)
    # (62.143760681152344, -1.690099835395813, -419.2769775390625), (328.66015625, -58.919437408447266, -195.18653869628906), (65.40367126464844, 400.0306701660156), (320, 240)
    # (119.03702545166016, -24.69566535949707, -239.9442596435547), (180.32456970214844, 221.3729248046875, -128.48776245117188), (-246.33840942382812, 203.94789123535156), (320, 240)
    # (140.0, -40.0, -220.0), (56.0, 0.0, 0.0), (52, 50), (320, 240)
    # (134.31761169433594, -28.56427764892578, -199.00991821289062), (54.66664505004883, 5.876393795013428, -10.042146682739258), (48.77881622314453, 50.049312591552734), (320, 240)
    # (142.76063537597656, -37.38973617553711, -222.53807067871094), (58.60441589355469, 2.5239651203155518, -1.2797986268997192), (54.54129409790039, 51.247108459472656), (320, 240)
    # (143.325439453125, -37.89916229248047, -217.80101013183594), (30.0211238861084, -26.496797561645508, 50.47840118408203), (55.39643478393555, -0.6100906729698181), (320, 240)
    # that is a lot of real shit up there!
    (161.36, -6, -235.4),
    (56.1, 1.0, -0.35),
    (51.17, 51.04),
    (320, 240),
)


def process(
        time: float, image: cv2.UMat | np.ndarray | None
) -> (
        tuple[
            float,
            list[tuple[float, float]],
            list[tuple[float, float]],
            list[tuple[tuple[float, float], tuple[float, float]]],
        ]
        | None
):
    """
    Process the given image to extract relevant information.

    Args:
        time (float): The time associated with the image.
        image (cv2.UMat | None): The image to be processed.

    Returns:
        tuple (tuple | None): A tuple containing the following information or None.
            - time (float): The time associated with the image.
            - reds (list): A list of red points found in the image.
            - yellows (list): A list of yellow points found in the image.
            - walls (list): A list of wall segments found in the image.
            - If the image is None, None is returned.
    """
    if image is None:
        return None

    mask_red, reds_in_image = find_color.find_red(image)
    mask_yellow, yellows_in_image = find_color.find_yellow(image)
    mask_blue, mask_white, walls_in_image = find_color.find_wall_bottom_p(image)

    if ENABLE_WEIRD_COLOR_DETECTING:
        mask_else = 255 - np.max(np.stack((mask_red, mask_yellow, mask_blue, mask_white), axis=0), axis=0)
        small_mask_else = block_or(mask_else, 10)
        ...  # TODO

    reds = []
    yellows = []
    walls = []

    for red in reds_in_image:
        s, x, y = camera_convert.img2space(CAMERA_STATE, red[0], red[1], -12.5)
        if s:
            reds.append((x, y))

    for yellow in yellows_in_image:
        s, x, y = camera_convert.img2space(CAMERA_STATE, yellow[0], yellow[1], -15)
        if s:
            yellows.append((x, y))

    if walls_in_image is not None:
        walls = [
            (
                camera_convert.img2space(CAMERA_STATE, wall[0][0], wall[0][1])[1:],
                camera_convert.img2space(CAMERA_STATE, wall[0][2], wall[0][3])[1:],
            )
            for wall in walls_in_image
        ]

    return time, reds, yellows, walls


def block_or(image: cv2.UMat | np.ndarray, block_size: int) -> np.ndarray:
    # 获取原始图像的尺寸
    h, w = image.shape
    # 新图像的尺寸
    new_h, new_w = h // block_size, w // block_size
    # 创建一个新图像
    new_image = np.zeros((new_h, new_w), dtype=np.uint8)

    # 对于每个块进行处理
    for i in range(new_h):
        for j in range(new_w):
            # 提取原图中的一个块
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            # 计算块中所有像素的“或”值
            new_image[i, j] = np.bitwise_or.reduce(block, axis=(0, 1))

    return new_image

    # size = image.shape[0] // 5, image.shape[1] // 5
    # output = np.full(size, 0, dtype=np.uint8)
    # for x in range(size[0]):
    #     for y in range(size[1]):
    #         flag = False
    #         for i in range(5):
    #             for j in range(5):
    #                 if image[5 * x + i, 5 * y + j] > 0:
    #                     flag = True
    #                     break
    #             if flag:
    #                 break
    #         if flag:
    #             output[x, y] = 255
    # return output
