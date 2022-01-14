# A set of functions to load images to python and then perform specific operstions on those images.
from PIL import Image
# The PIL library was not installing via pycharm and I had to execute pip3 on the terminal to install it.
# Using Pillow which is a form of PIL
import numpy as np

def import_local_image(path, resize = False, rhx = 400, rwx = 400):
    '''
    Imports a local image and then returns an image object.
    :param path:
    :return: a tuple containing both the image as well as the numpy representation of the image as an array
    '''
    image = Image.open(path)
    # when a specific image is converted into an np array it is stored as an np array of shape
    # height X width X layers.
    # However when using the resize method in an image object we specify the target size as width x height.
    # so a h= 1000, w = 300 image to be resized in to a h = 400, w = 200 image, the method will be executed with
    # the tuple (200,400) as the input.
    # But on converting into an np array we will get the shape as 400X200X3
    if resize:
        resized_image = image.resize((rwx, rhx))
        array_form = np.array(resized_image)
    else:
        array_form = np.array(image)
    return image, array_form