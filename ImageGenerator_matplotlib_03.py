from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

nb_class = 10
nb_class_list = list(range(nb_class))

(IMG_W, IMG_H, IMG_D) = (28, 28, 1)
img_list = []

PATH = '/Users/ichiroamitani/Documents/Software/Python/Siamese_network-master/image/'

print("Image Read Started")
for i in nb_class_list:

    for filename in glob.glob(PATH + 'train/'  + str(i) + '/*.jpg'):
        img = Image.open(filename).convert('L')
        img = np.asarray(img)/255
        img_list.append(img)

print("Image Read Finished")


def show_effect(img_list, x, grid_shape = (3, 3)):

    (r, c) = grid_shape
    fig, axes = plt.subplots(r, c)
    k = 0
    for i in range(c):
        for j in range(r):
            axes[i, j].matshow(img_list[k].reshape(IMG_W, IMG_H), cmap = 'gray')
            axes[i, j].get_yaxis().set_visible(False)
            axes[i, j].get_xaxis().set_visible(False)
            k = k + 1

    plt.show()


def show_effect_2(img_list, x, grid_shape = (3, 3)):

    (r, c) = grid_shape
    gs = gridspec.GridSpec(r, c)
    gs.update(wspace = 0.1, hspace = 0.1)

    for i in range(r * c):

        plt.subplot(gs[i])
        plt.imshow(img_list[i].reshape(IMG_W, IMG_H), cmap = 'gray', aspect = 'auto')
        plt.axis("off")

    plt.show()


def generate_augmented_image(data_generator, img_src, nb_images = 9):

    g = datagen.flow(img_src, batch_size = 1)

    img_list = []
    for i in range(nb_images):
        img_list.append(g.next())

    return img_list


if __name__ == '__main__':

    x = img_list[0].reshape(1, IMG_H, IMG_W, 1)
    grid_shape = (3, 3)

    datagen = ImageDataGenerator(rotation_range = 90)
    img_list = generate_augmented_image(datagen, img_src = x)

    with open('./rotated_images.pickle','wb') as f:
        pickle.dump(img_list, f)

    with open('./rotated_images.pickle','rb') as f:
        image_list = pickle.load(f)

    show_effect_2(image_list, x)

#
#
#
