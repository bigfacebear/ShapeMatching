#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from random import randint

import numpy as np
import sys
from PIL import Image, ImageDraw

import nshapegenflags



DIM = nshapegenflags.DIM  # Shorter
COLOR = nshapegenflags.COLOR
RANDOM_COLOR = nshapegenflags.RANDOM_COLOR


def set_dim(dim):
    nshapegenflags.DIM = dim


def set_color(color):
    nshapegenflags.COLOR = color


def generate_image_pairs(n):
    print_progress_bar(0, n, prefix="Generating images: ", suffix="Done!", decimals=2, length=100)
    for i in range(1, n + 1):
        print_progress_bar(i, n, prefix="Generating images: ", suffix="Done!", decimals=2, length=100)
        save_image_pair(randint(0, 2), i)


def save_image_pair(shape, id):
    (top, bottom) = get_image_pair(shape)

    bottom.save("images/" + str(id) + "_L.png")
    top.save("images/" + str(id) + "_K.png")


def get_image_pair(shape):
    im = get_shape_image(shape)

    # Crop the top part of the image
    top = im.crop((0, 0, DIM, DIM / 2))
    top.load()

    # Crop the bottom part of the image
    bottom = im.crop((0, DIM / 2, DIM, DIM))
    bottom.load()

    # Pad the images to 100x100
    top = full_size(top)
    bottom = full_size(bottom)

    # Rotate randomly
    if nshapegenflags.ROTATE:
        top = top.rotate(random_angle_degrees(0 - nshapegenflags.ROTATE_MAX_DEGREES, nshapegenflags.ROTATE_MAX_DEGREES))
        bottom = bottom.rotate(random_angle_degrees(0 - nshapegenflags.ROTATE_MAX_DEGREES, nshapegenflags.ROTATE_MAX_DEGREES))

    return (top, bottom)


# Based on https://stackoverflow.com/a/11143078
def full_size(image, dimensions=(DIM, DIM)):
    new_size = dimensions
    new_im = Image.new("RGB", new_size)  # Black by default
    new_im.paste(image, ((new_size[0] - image.size[0]) / 2,
                          (new_size[1] - image.size[1]) / 2))

    return new_im

def get_shape_image(shape):
    im = Image.new('RGB', (DIM, DIM))

    if shape == 0:
        im = ellipse(im)
    elif shape == 1:
        im = triangle(im)
    elif shape == 2:
        im = square(im)


    return im



def get_color():
    return random_color() if RANDOM_COLOR else COLOR



def square(im, size=0.6 * DIM):
    # Square in the center of the image
    vertices = [(DIM / 2 - size / 2, DIM / 2 - size / 2), (DIM / 2 - size / 2, DIM / 2 + size / 2),
                (DIM / 2 + size / 2, DIM / 2 + size / 2), (DIM / 2 + size / 2, DIM / 2 - size / 2)]

    # vertices = rotate(vertices, random_angle())

    # Draw it on the image
    draw = ImageDraw.Draw(im)
    color = get_color()
    draw.polygon(vertices, fill=color, outline=color)
    del draw

    im = im.rotate(random_angle_degrees())

    return im


def triangle(im, size=0.6 * DIM):
    # Triangle in the center of the image
    vertices = [(DIM / 2 - size / 2, DIM / 2 + size * math.sqrt(3) / 4),
                (DIM / 2 + size / 2, DIM / 2 + size * math.sqrt(3) / 4), (DIM / 2, DIM / 2 - size * math.sqrt(3) / 4)]

    # vertices = rotate(vertices, random_angle())

    # Draw it on the image
    draw = ImageDraw.Draw(im)
    color = get_color()
    draw.polygon(vertices, fill=color, outline=color)
    del draw

    im = im.rotate(random_angle_degrees())

    return im


def ellipse(im):
    # Generate random bounds of ellipse; avoid having ellipses that are too narrow
    e_x = randint(math.floor(0.2 * DIM), math.floor(0.8 * DIM))
    e_y = randint(DIM - e_x, 0.8 * DIM)

    # Draw ellipse
    bbox = (DIM / 2 - e_x / 2, DIM / 2 - e_y / 2, DIM / 2 + e_x / 2, DIM / 2 + e_y / 2)
    draw = ImageDraw.Draw(im)
    color = get_color()
    draw.ellipse(bbox, fill=color, outline=color)
    del draw

    im = im.rotate(random_angle_degrees())

    return im


def rotate(points, angle):
    result = list()

    # Generate rotation matrix based on angle
    rotation_matrix = np.matrix([[math.cos(angle), 0 - math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    # Loop through every point in the list of vertices
    for point in points:
        # Create a vector out of the point: (while setting center to origin)
        #   [[x]
        #    [y]]
        point_vector = np.transpose(np.matrix([point[0] - DIM / 2, point[1] - DIM / 2]))

        # Multiply the rotation matrix by the point vector
        rotated = np.matmul(rotation_matrix, point_vector)

        # Reshape matrix into coordinate pair (and move origin back to origin from center)
        rotated_vertices = (math.floor(rotated.item((0, 0))) + DIM / 2, math.floor(rotated.item((1, 0))) + DIM / 2)

        # Add rotated vertex to list
        result.append(rotated_vertices)

    return result


def random_angle():
    return randint(0, 359) * math.pi / 180  # randint is inclusive, so 0 to 359

def random_color():
    return randint(0, 255), randint(0, 255), randint(0, 255)  # Inclusive

def random_angle_degrees(min=0, max=359):
    return randint(min, max - 1)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar. Based on https://stackoverflow.com/a/34325723
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix) + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()
        print()
        sys.stdout.write("")

