import math
from random import randint

import numpy as np
from PIL import Image, ImageDraw

import Flags



def shape(id, which):
    name = ""

    im = Image.new('RGB', (Flags.DIM, Flags.DIM))

    while not stable(im):

        im = Image.new('RGB', (Flags.DIM, Flags.DIM))

        shapeName = ""

        if which == 0:
            im = ellipse(im)
            shapeName = "ellipse"
        elif which == 1:
            im = triangle(im)
            shapeName = "triangle"
        elif which == 2:
            im = square(im)
            shapeName = "square"

        im = cut(im)

        name = "images/" + str(id) + "." + shapeName + ".png"
        # name = str(id) + "." + shapeName

    # return (im, name)
    im.save(name)
    # im.show()


def ellipse(im, xBoundMin=math.floor(0.2 * Flags.DIM), xBoundMax=math.floor(0.8 * Flags.DIM), yBoundMin=math.floor(0.2 * Flags.DIM), yBoundMax=math.floor(0.8 * Flags.DIM)):
    # Generate random bounds of ellipse based on inputs
    x, y = im.size
    eX, eY = randint(xBoundMin, xBoundMax), randint(yBoundMin, yBoundMax)

    # Draw ellipse
    bbox = (x / 2 - eX / 2, y / 2 - eY / 2, x / 2 + eX / 2, y / 2 + eY / 2)
    draw = ImageDraw.Draw(im)
    draw.ellipse(bbox, fill=128, outline=128)
    del draw

    return im


def triangle(im, size=0.78 * Flags.DIM):
    # Generate a roughly equilateral triangle
    size = Flags.DIM - size
    vertices = [(size, Flags.DIM - size), (Flags.DIM - size, Flags.DIM - size), (Flags.DIM / 2, size)]

    # Rotate it anywhere from 0 to 2pi radians
    vertices = rotate(vertices, randint(0, 360) * math.pi / 180)

    # Draw it on the image
    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=128, outline=128)
    del draw

    return im


def square(im, size=0.2 * Flags.DIM):
    # Generate a square
    vertices = [(Flags.DIM / 2 - size, Flags.DIM / 2 - size), (Flags.DIM / 2 - size, Flags.DIM / 2 + size), (
    Flags.DIM / 2 + size, Flags.DIM / 2 + size), (Flags.DIM / 2 + size, Flags.DIM / 2 - size)]

    # Rotate it anywhere from 0 to 2pi radians
    vertices = rotate(vertices, randint(0, 360) * math.pi / 180)

    # Draw it on the image
    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=128, outline=128)
    del draw

    return im


def rotate(points, angle):
    result = list()

    rotation_matrix = np.matrix([[math.cos(angle), 0 - math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    for point in points:
        point_vector = np.transpose(np.matrix([point[0] - Flags.DIM / 2, point[1] - Flags.DIM / 2]))

        rotated = np.matmul(rotation_matrix, point_vector)

        rotated_vertices = (math.floor(rotated.item((0, 0))) + Flags.DIM / 2, math.floor(rotated.item((1, 0))) + Flags.DIM / 2)

        result.append(rotated_vertices)

    return result


def cut(im):
    return smarterCut(im)


def simpleCircleDefinedLineCut(im, r=0.3 * Flags.DIM):
    # TODO: Make it work for both positive and negative signs
    # FIXME (this function does not work)

    # Pick points a and b on the circle, but not too close to the boundary
    a = randint(Flags.DIM / 2 - r, Flags.DIM / 2 + r)
    b = Flags.DIM / 2 + math.sqrt(r ** 2 - (a - Flags.DIM / 2) ** 2)

    # slope (based on derivative)
    m = 0 - ((a - Flags.DIM / 2) / (math.sqrt(r ** 2 - (a - Flags.DIM / 2) ** 2)))

    # y-intercept value
    yInt = -a * m + b

    # x100-intercent value
    x100Int = (0 - (Flags.DIM / 2) * (math.sqrt(0 - a ** 2 + Flags.DIM * a + r ** 2 - (
    Flags.DIM / 2) ** 2) + Flags.DIM / 2) + Flags.DIM / 2 * a + r ** 2) / (a - Flags.DIM / 2)

    if (x100Int > Flags.DIM):  # Bound the x100-intercept at 100
        x100Int = Flags.DIM

    print x100Int
    print yInt
    print "___"
    vertices = [(0, 0), (0, x100Int), (Flags.DIM - yInt, 0)]

    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=128, outline=128)
    del draw

    return im


def simpleLineCut(im):
    '''Trivial line generation--simply chooses two boundary points, one on top, one on bottom.'''
    top = randint(1, Flags.DIM)
    bottom = randint(1, Flags.DIM)

    vertices = [(0, 0), (randint(1, top), 0), (randint(1, bottom), Flags.DIM), (0, Flags.DIM)]

    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=0, outline=0)
    del draw

    return im


def smarterCut(im):
    (x, y) = (0, 0)
    good = False

    # Pick a random pair of pixles within the shape
    while good == False:
        x = randint(0, Flags.DIM - 1)
        y = randint(0, Flags.DIM - 1)

        (r, g, b) = im.getpixel((x, y))
        if r != 0:
            good = True

    qs = randint(0, 1)
    q = 0
    if qs == 1:
        q = 1
    else:
        q = -1

    m = randint(1, 1000) * 0.001

    y_intercept = math.floor(0 - x * q * math.log(m) + y)
    if y_intercept < 0:
        y_intercept = 0
    elif y_intercept > Flags.DIM:
        y_intercept = Flags.DIM

    qr = randint(0, 1)

    vertices = []
    if qr == 0:
        vertices = [(0, 0), (Flags.DIM, 0), (Flags.DIM, randint(0, y_intercept)), (0, y_intercept)]
    else:
        vertices = [(0, Flags.DIM), (0, y_intercept), (Flags.DIM, randint(0, Flags.DIM - y_intercept)), (
        Flags.DIM, Flags.DIM)]

    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=0, outline=0)
    del draw

    return im


def stable(im):
    '''Stability checks on the generated images (especially needed for naive cutting algorithms)'''

    valid = True

    num_colored = 0

    for i in range(0, Flags.DIM - 1):
        for j in range(0, Flags.DIM - 1):
            (r, g, b) = im.getpixel((i, j))

            if (r == 128):
                num_colored = num_colored + 1

    percentColored = num_colored * 100 / Flags.DIM ** 2

    if (percentColored < 5):
        valid = False

    return valid
