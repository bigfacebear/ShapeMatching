import os

import nshapegen
import nshapegenflags


# Create directory for storing images
if not os.path.exists("images"):
    os.makedirs("images")

nshapegen.generate_image_pairs(nshapegenflags.IMAGE_NUM)
