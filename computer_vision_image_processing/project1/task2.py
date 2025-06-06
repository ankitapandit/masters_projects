"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *   # you could modify this line

template_image_name = ""


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/proj1-task2.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    args = parse_args()
    template_image_name = args.template_path
    norm_img = normalize(img)
    norm_template = normalize(template)
    img_row_zero = len(img)
    img_pixel_zero = len(img[0])
    template_row_zero = len(template)
    template_pixel_zero = len(template[0])
    coordinates = list()

    threshold = 0.975
    if template_image_name == "./data/a.jpg":
        threshold = 0.982
    elif template_image_name == "./data/b.jpg":
        threshold = 0.977
    elif template_image_name == "./data/c.jpg":
        threshold = 0.977

    for i in range(1, img_row_zero):
        for j in range(1, img_pixel_zero):
            p, q, r = 0, 0, 0
            skip = False
            for k in range(1, template_row_zero):
                for l in range(1, template_pixel_zero):
                    x = i + k - 1
                    y = j + l - 1
                    if (x >= img_row_zero or y >= img_pixel_zero):
                        skip = True
                        break
                    p += float(norm_img[x][y] * norm_template[k][l])
                    q += float(norm_img[x][y] ** 2)
                    r += float(norm_template[k][l] ** 2)
            if not skip:
                norm_cross_correlation = float(p / np.sqrt(q * r))
                flag = is_valid(norm_cross_correlation, threshold)
                if flag:
                    coordinate = [i, j]
                    coordinates.append(coordinate)
                flag = False
    return coordinates


def is_valid(value, threshold):
    if value >= threshold:
        return True
    else:
        return False


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    img = read_image(args.img_path)
    template = read_image(args.template_path)
    coordinates = detect(img, template)
    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
