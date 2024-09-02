import os.path
import time

import cv2
import numpy as np

from communication.camera import Camera

try:
    from algorithm import camera_convert, find_color, vision_test
except ModuleNotFoundError:
    import os
    import sys

    current_dir = os.path.dirname(__file__)
    sys.path.append(f"{current_dir}")
    from algorithm import camera_convert, find_color, vision_test

else:
    ...

"""
MODE = "camera", "file", "adjust"
"""
MODE = "adjust"

FORCE_OVERWRITE = False
GLOBAL_SHOW = True
MASK_SHOW = True
READ_DIR = "color_dot_far"
# naming rule: forward dist first, right dist second
WRITE_DIR = "cars2"

# CAMERA_STATE = camera_convert.CameraState((269, 1, -178), (90 - 29.8, 2.0, 0.2), (62.2, 48.8), (640, 480))
# CAMERA_STATE = camera_convert.CameraState((286, 2, -197), (90 - 33.3, 2.0, 0.0), (62.2, 55), (640, 480))
# CAMERA_STATE = camera_convert.CameraState((303, 0, -212), (53.7, 2.0, 0.4), (62.2, 60), (640, 480))
# CAMERA_STATE = camera_convert.CameraState(
#     (269, 12, -198), (53.2, 0.9, 0.9), (62.2, 62), (640, 480)
# )
# (357, 10, -207) [51.7  1.2  1. ]
# (295, 12, -221) [57.7  1.1  0.3] (np.float64(51.44557864216593), np.float64(51.08994556912829))
CAMERA_STATE = vision_test.CAMERA_STATE

SHOW_RED = SHOW_YELLOW = True
SHOW_WALL = True
DRAW_GRID = True

USE_HOUGH_P = True


def process(img, show: bool = False, img_=None):
    mask_red, points_red = find_color.find_red(img, show and MASK_SHOW)
    mask_yellow, points_yellow = find_color.find_yellow(img, show and MASK_SHOW)
    mask_blue, mask_white, walls = (
        find_color.find_wall_bottom_p(img, show and MASK_SHOW)
        if USE_HOUGH_P
        else find_color.find_wall_bottom(img, show and MASK_SHOW)
    )
    # mask_else = 255 - np.max(np.stack((mask_red, mask_yellow, mask_blue, mask_white), axis=0), axis=0)

    mask_else = 255 - np.max(np.stack((mask_red, mask_yellow, mask_blue, mask_white), axis=0), axis=0)
    small_mask_else = vision_test.block_or(mask_else, 10)
    if show:
        cv2.imshow('else', small_mask_else)
    ...  # TODO

    # switch to a new line
    print()

    to_draw = img if img_ is None else img_
    y_shift = 0 if img_ is None else img_.shape[0] - img.shape[0]
    print(y_shift)

    for p in points_red:
        s, x, y = camera_convert.img2space(CAMERA_STATE, p[0], p[1], -12.5)
        if s:
            if show and SHOW_RED:
                cv2.rectangle(to_draw, (p[0] - 10, p[1] + y_shift - 10, 20, 20), (255, 255, 255, 255), 2)
            print((x, y), "red")
        else:
            if SHOW_RED:
                cv2.rectangle(to_draw, (p[0] - 10, p[1] + y_shift - 10, 20, 20), (128, 128, 128, 255), 1)

    for p in points_yellow:
        s, x, y = camera_convert.img2space(CAMERA_STATE, p[0], p[1], -15)
        if s:
            if show and SHOW_YELLOW:
                cv2.circle(to_draw, (p[0], p[1] + y_shift), 10, (255, 255, 255, 255), 2)
            print((x, y), "yellow")
        else:
            if show and SHOW_YELLOW:
                cv2.circle(to_draw, (p[0], p[1] + y_shift), 10, (128, 128, 128, 255), 1)

    if show and DRAW_GRID:
        draw_grid(to_draw, (255, 255, 255, 255), 0, 2000, 200, -400, 400, 200, y_shift)
        draw_grid(to_draw, (255, 255, 255, 255), 2000, 102000, 100000, -500, 500, 50, y_shift)

    if walls is not None:
        for w in walls:
            if USE_HOUGH_P:
                h1, v1, h2, v2 = w[0]
            else:
                rho, theta = w[0]
                a = np.cos(theta)
                b = np.sin(theta)
                h0 = a * rho
                v0 = b * rho
                h1 = int(h0 + 1000 * (-b))
                v1 = int(v0 + 1000 * a)
                h2 = int(h0 - 1000 * (-b))
                v2 = int(v0 - 1000 * a)
            if show and SHOW_WALL:
                cv2.line(to_draw, (h1, v1 + y_shift), (h2, v2 + y_shift), (255, 255, 255, 255), 1)
            s1, x1, y1 = camera_convert.img2space(CAMERA_STATE, h1, v1, 0)
            s2, x2, y2 = camera_convert.img2space(CAMERA_STATE, h2, v2, 0)
            print(((x1, y1), (x2, y2)), "wall")

    if show:
        # overlay = np.repeat(mask_else[:, :, np.newaxis], 3, axis=2)
        #
        # cv2.add(overlay, img, img)

        cv2.imshow("image", to_draw)
        print("no shit")


def draw_grid(img, color, x_start, x_stop, x_step, y_start, y_stop, y_step, y_shift: int = 0):
    overlay = np.zeros(img.shape, np.uint8)

    for x in range(x_start, x_stop + x_step, x_step):
        for y in range(y_start, y_stop, y_step):
            s1, i1, j1 = camera_convert.space2img(CAMERA_STATE, x, y)
            s2, i2, j2 = camera_convert.space2img(CAMERA_STATE, x, y + y_step)
            # if s1 or s2:
            cv2.line(overlay, (i1, j1 + y_shift), (i2, j2 + y_shift), color if x != 0 else (0, 0, 255), 1)
    for y in range(y_start, y_stop + y_step, y_step):
        for x in range(x_start, x_stop, x_step):
            s1, i1, j1 = camera_convert.space2img(CAMERA_STATE, x, y)
            s2, i2, j2 = camera_convert.space2img(CAMERA_STATE, x + x_step, y)
            # if s1 or s2:
            cv2.line(overlay, (i1, j1 + y_shift), (i2, j2 + y_shift), color if y != 0 else (0, 0, 255), 1)

    overlay = np.minimum(
        overlay,
        np.repeat((255 - find_color.get_color_mask(img, [find_color.RED1]))[:, :, np.newaxis], 3, axis=2),
    )
    overlay = np.minimum(
        overlay,
        np.repeat((255 - find_color.get_color_mask(img, [find_color.RED2]))[:, :, np.newaxis], 3, axis=2),
    )
    overlay = np.minimum(
        overlay,
        np.repeat((255 - find_color.get_color_mask(img, [find_color.YELLOW]))[:, :, np.newaxis], 3, axis=2),
    )
    overlay = np.minimum(
        overlay,
        np.repeat((255 - find_color.get_color_mask(img, [find_color.BLUE]))[:, :, np.newaxis], 3, axis=2),
    )

    cv2.add(overlay, img, img)


if __name__ == "__main__":

    repository_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
    if MODE == "file":
        for image_index in range(100):
            filename = repository_path + "/assets/openCV_pic/" + READ_DIR + "/" + str(image_index) + ".jpg"
            if not os.path.isfile(filename):
                print("cannot open " + filename)
                continue
            image = cv2.imread(filename)

            process(image, True)
            cv2.waitKey()

    if MODE == "adjust":
        filename = repository_path + "/assets/openCV_pic/" + READ_DIR + "/0.jpg"
        filename_ = repository_path + "/assets/openCV_pic/" + READ_DIR + "/0_.jpg"
        if not os.path.isfile(filename):
            print("cannot open " + filename)
            exit(0)
        else:
            image = cv2.imread(filename)
        if not os.path.isfile(filename_):
            print("cannot open " + filename_)
            image_ = None
        else:
            image_ = cv2.imread(filename_)
        while True:
            img_temp = image.copy()
            img__temp = image_.copy() if image_ is not None else None
            process(img_temp, True, img__temp)
            key = cv2.waitKey()
            print(key)
            if key == ord("a"):
                CAMERA_STATE.phi += np.radians(0.1)
            elif key == ord("d"):
                CAMERA_STATE.phi += np.radians(-0.1)
            elif key == ord("w"):
                CAMERA_STATE.theta += np.radians(-0.1)
            elif key == ord("s"):
                CAMERA_STATE.theta += np.radians(0.1)
            elif key == ord("q"):
                CAMERA_STATE.omega += np.radians(0.1)
            elif key == ord("e"):
                CAMERA_STATE.omega += np.radians(-0.1)
            elif key == ord("i"):
                CAMERA_STATE.z += 1
            elif key == ord("k"):
                CAMERA_STATE.z += -1
            elif key == ord("u"):
                CAMERA_STATE.x += -1
            elif key == ord("j"):
                CAMERA_STATE.x += 1
            elif key == ord("o"):
                CAMERA_STATE.y += 1
            elif key == ord("p"):
                CAMERA_STATE.y += -1
            elif key == ord("n"):
                CAMERA_STATE.half_fov_h += 0.001
            elif key == ord("m"):
                CAMERA_STATE.half_fov_h += -0.001
            elif key == ord("v"):
                CAMERA_STATE.half_fov_v += 0.001
            elif key == ord("b"):
                CAMERA_STATE.half_fov_v += -0.001
            CAMERA_STATE.update()
            print(
                (CAMERA_STATE.x, CAMERA_STATE.y, CAMERA_STATE.z),
                np.degrees((CAMERA_STATE.theta, CAMERA_STATE.phi, CAMERA_STATE.omega)),
                (CAMERA_STATE.half_fov_h / np.pi * 360, CAMERA_STATE.half_fov_v / np.pi * 360),
            )

    if MODE == "camera":
        target_dir = repository_path + "/assets/openCV_pic/" + WRITE_DIR + "/"
        if os.path.isdir(target_dir):
            if not FORCE_OVERWRITE:
                print("Directory", target_dir, "exists. Consider enabling FORCE_OVERWRITE.")
                exit(0)
        else:
            os.mkdir(target_dir)
        c = Camera()
        time_last_capture = time.time() + 1.5
        for image_index in range(100):
            print(int((time.time() - time_last_capture) * 1000))
            time.sleep(max(time_last_capture + 0.5 - time.time(), 0))
            image = c.get_image_bgr()
            time_last_capture = time.time()
            filename = target_dir + str(image_index) + ".jpg"

            if image is not None:
                cv2.imwrite(filename, image)
                process(image, GLOBAL_SHOW)

            else:
                print("shit")
            cv2.waitKey(200)
