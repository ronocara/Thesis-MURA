import MURA
import cv2
import image_manipulation
from multiprocessing import Pool
from p_tqdm import p_map
import itertools
from models import Autoencoder
from os import listdir
import numpy as np
import glob
from math import ceil, floor
from os import path, mkdir
from argparse import ArgumentParser



# All editable variables
image_paths = MURA.MURA_DATASET()
new_file_path = "MURA-v1.1/augmented/test_1"
all_image_paths = image_paths.get_combined_image_paths().to_numpy()
# Numpy random seed for dataset shuffling
np.random.seed(15)
# Dataset Split - Should sum up to 1.0
training_set = 0.4
validation_set = 0.05
testing_set = 0.55


# Augmentation parameters
training_augmentation = {'augment_hflip': False,
 'augment_vflip': False,
 'max_rotation': 0}

testing_augmentation = {'augment_hflip': False,
 'augment_vflip': False,
 'max_rotation': 0}

validation_augmentation = {'augment_hflip': False,
 'augment_vflip': False,
 'max_rotation': 0}

augmentations = [training_augmentation, validation_augmentation, testing_augmentation]

# max threads to be used
num_processes = 8
# num of epochs
epochs = 1
# num of batch size
batch_size = 64

# Applies various data_cleaning methods chosen from data_cleaning.py
def apply_data_cleaning(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_manipulation.adaptive_histogram(image)
    image = image_manipulation.watershed(image)
    # image = image_manipulation.black_and_white(image)
    image = image_manipulation.resize(image)
    return image

# Iterate on image paths and apply data cleaning processes
# Saves images on new_file_path and gets images from image_paths
# image_paths should be a tuple containing (integer, string of path, label)
def process_image(new_file_path, image_path, augmentation_data):

    image = cv2.imread(image_path)
    image = apply_data_cleaning(image)
    new_train_file_path = f"{new_file_path}/{image_path.replace('/','-').replace('.png', '')}"

    #augments the data
    new_images = image_manipulation.augment_data(image, hflip= augmentation_data['augment_hflip'], 
                                                                                 vflip=augmentation_data['augment_vflip'],
                                                                                 max_rotation=augmentation_data['max_rotation'])
    for index, img in enumerate(new_images):
        cv2.imwrite(f"{new_train_file_path}_{index}.png", img) 
    return

def create_dir(parent_dir, new_dir_name):
    if path.isdir(f'{parent_dir}/{new_dir_name}'):
        return
    else:
        mkdir(f'{parent_dir}/{new_dir_name}')

def data_preparation():
    print("Shuffling Dataset\n")
    np.random.shuffle(all_image_paths)

    print("Splitting Dataset\n")
    total_images = len(all_image_paths)
    positives = []
    negatives = []
    for image_path in all_image_paths:
        if image_path[1] == 1.0:
            positives.append(image_path[0])
        else:
            negatives.append(image_path[0])

    training = []
    validation = []
    testing = []
    
    for i in range(ceil(total_images*training_set)):
        training.append(negatives.pop())
    for i in range(floor(total_images*validation_set)):
        validation.append(negatives.pop())
    for i in range(len(positives)):
        testing.append(positives.pop())
    for i in range(len(negatives)):
        testing.append(negatives.pop())

    np.random.shuffle(testing)
    datasets = [training, validation, testing]
    dir_names = ['training', 'validation', 'testing']
    for i in range(3):
        create_dir(new_file_path, dir_names[i])
        print(f"Processing {dir_names[i]} images\n")
        with Pool(num_processes) as p:
            p_map(process_image, itertools.repeat(f'{new_file_path}/{dir_names[i]}', len(datasets[i])), 
                                               datasets[i], 
                                               itertools.repeat(augmentations[i], len(datasets[i])))
        print("\n")


    # each array contains the training, validation, and testing in order
    image_datasets = [[],[],[]]
    for i in range(3):
        for image_path in glob.glob(f'{new_file_path}/{dir_names[i]}/*.png'):
            image_datasets[i].append(cv2.imread(image_path))
        image_datasets[i] = np.array(image_datasets[i])

    return image_datasets;

if __name__ == "__main__":

    #for either VAE or UPAE
    parser = ArgumentParser()
    parser.add_argument('--u', dest='u', action='store_true') # use uncertainty
    opt = parser.parse_args()

    #preprocessing and augmentation
    image_datasets = data_preparation()

    # Creates and trains the model
    multiplier = 4
    latentSize= 16
    input_shape=(64,64,3)

    print("Training AE model")
    vae = Autoencoder(input_shape, multiplier, latentSize, upae=opt.u)
    vae.compile_AE(upae=opt.u)
    model = vae.fit_AE(image_datasets[0], 
                       image_datasets[0],
                       x_test=image_datasets[1], #adding validation data for tensorboard 
                       epochs=epochs,
                       batch_size=batch_size,
                       )
    
    #validate on validation set (normal images)
    print("Validating AE Model")
    score = vae.validate(image_datasets[1], batch_size=32, upae=opt.u)

    #test on an image
    #print("Testing AE Model")
    

    #get reconstruction error using MSE

