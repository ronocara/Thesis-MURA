import numpy as np
import glob
import cv2
import os

quantization_levels = 16
source_folder = "dataset_2"
output_folder = "dataset_2_quantization"


# each array contains the training, validation, and testing in order
image_datasets = {'train': [], 'test': [], 'valid': []}
for dataset_name in image_datasets.keys():
    os.makedirs(f'datasets/{output_folder}/{dataset_name}', exist_ok=True)
    save_path = f'datasets/{output_folder}/{dataset_name}/'
    print(f'{source_folder}/{dataset_name}/*.png')
    for image_path in glob.glob(f'datasets/{source_folder}/{dataset_name}/*.png'):
        image = cv2.imread(image_path)
        min_value = np.min(image)
        max_value = np.max(image)

        step = (max_value - min_value) / quantization_levels
        boundaries = np.linspace(min_value, max_value, quantization_levels + 1)


        quantized_image = np.digitize(image, boundaries) - 1  # Subtract 1 to get levels from 0 to 15

        # Scale the quantized image back to the original range (0-255)
        quantized_image = quantized_image * (255 / (quantization_levels - 1))

        # Convert the quantized image to unsigned 8-bit integers
        quantized_image = quantized_image.astype(np.uint8)

        image_path = image_path.split("\\")[-1]
        image_path_name = save_path+image_path
        cv2.imwrite(save_path+image_path, quantized_image)



