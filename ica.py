from matplotlib import pyplot as plt
from skimage import data, io, color
from sklearn.decomposition import FastICA

import multiprocessing
import numpy as np


def create_noisy_img(img_a, img_b, alpha=0.5):
    return img_a * alpha + (1 - alpha) * img_b


def do_ica(noisy_imgs):
    ica = FastICA(n_components=2)
    ica.fit(noisy_imgs)

    restored_imgs = ica.inverse_transform(ica.transform(noisy_imgs))

    # Select the image with the middle alpha value
    restored_img = restored_imgs[len(restored_imgs) // 2]

    restored_img = np.array(restored_img).clip(0, 1).reshape((img_height, img_width))
    restored_img = np.hstack(
        (restored_img, np.ones([img_height, img_width]) - restored_img)
    )

    io.imsave("restored_img.jpg", restored_img)
    io.imshow(restored_img)
    plt.show()

    return restored_imgs


if __name__ == "__main__":
    img_a = io.imread("input/image-a.jpg", as_gray=True)
    img_b = io.imread("input/image-b.jpg", as_gray=True)
    img_height = len(img_a)
    img_width = len(img_a[0])
    noisy_imgs = [
        np.array(create_noisy_img(img_a, img_b, alpha)).ravel()
        for alpha in np.linspace(0.1, 0.9, 9)
        # for alpha in [0.25, 0.75]
    ]

    do_ica(noisy_imgs)
