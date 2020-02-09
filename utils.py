import pathlib

import matplotlib.pyplot as plt
from PIL import Image


def load_image(image_path):
    if not image_path:
        print('Please provide a valid image path ..')
    if not isinstance(image_path, pathlib.PurePath):
        image_path = pathlib.Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError
    else:
        return Image.open(image_path)


def map_plotter(grad_image, original_image):
    _, axs = plt.subplots(1, 2)

    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')

    axs[1].imshow(grad_image)
    axs[1].set_title('Importance map')
    plt.show()
