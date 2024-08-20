import cv2
import os
import shutil
import numpy as np

def Is_BlackAndWhite(image):
    
    b , g , r  = cv2.split(image)
    tolerance = 10
    return np.allclose(b , g , atol=tolerance) and np.allclose(g , r ,atol=tolerance)

def remove_black_and_white_images(dataset_path, output_path):
       
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # List all files in the dataset directory
    image_files = [f for f in os.listdir(dataset_path)]
    
    for file in image_files:
        file_path = os.path.join(dataset_path, file)
        image = cv2.imread(file_path)
        
        # Calculate the color difference
        diff = Is_BlackAndWhite(image)
        
        # Check if the difference is below the threshold (considered grayscale)
        if diff :
            print(f"Skipping grayscale image: {file}")
        else:
            # Copy color image to output directory
            new_file_path = os.path.join(output_path, file)
            shutil.copy(file_path, new_file_path)  # Copy the file
            print(f"Copying color image: {file} to {output_path} folder")

