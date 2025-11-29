# this script should create a tensor of processed images and store it.
# the shape should be (num_image_pairs, 2, 3, H, W)

import os
import torch
import numpy as np
from imageio import imread
import json

def process_and_save_images(input_image_dir, save_dir, captions_file):
    mean = [100.6790,  99.5023,  84.9932]
    std = [50.9820, 48.4838, 44.7057]
    splits = ["train", "val", "test"]
    
    for split in splits:
        input_image_dir_split = os.path.join(input_image_dir, split)
        save_dir_split = os.path.join(save_dir, split)
        if not os.path.exists(save_dir_split):
            os.makedirs(save_dir_split)
            
        levir_captions = json.load(open(captions_file, 'r'))
        images = levir_captions['images']
        
        # Count images first to preallocate
        num_images = sum(1 for img in images if img["split"] == split)
        
        # Get image dimensions from first image
        sample_img = None
        for img_info in images:
            if img_info["split"] == split:
                sample_path = os.path.join(input_image_dir_split, 'A', img_info["filename"])
                sample_img = imread(sample_path)
                break
        
        H, W = sample_img.shape[:2]
        print(f"Processing {split}: Found {num_images} images of size ({H}, {W})")
        
        # Preallocate tensor with correct dtype
        images_tensor = torch.zeros((num_images, 2, 3, H, W), dtype=torch.float32)
        
        idx = 0
        for img_info in images:
            if img_info["split"] == split:
                img_filename = img_info["filename"]
                img_path_A = os.path.join(input_image_dir_split, 'A', img_filename)
                img_path_B = os.path.join(input_image_dir_split, 'B', img_filename)
                
                imgA = imread(img_path_A).astype(np.float32)
                imgB = imread(img_path_B).astype(np.float32)
                
                imgA = np.moveaxis(imgA, -1, 0)     
                imgB = np.moveaxis(imgB, -1, 0)
                
                # Normalize in-place
                for i in range(3):
                    imgA[i] = (imgA[i] - mean[i]) / std[i]
                    imgB[i] = (imgB[i] - mean[i]) / std[i]
                
                # Write directly to preallocated tensor
                images_tensor[idx, 0] = torch.from_numpy(imgA)
                images_tensor[idx, 1] = torch.from_numpy(imgB)
                idx += 1
        
        torch.save(images_tensor, os.path.join(save_dir_split, 'processed_images.pt'))
        print(f"Saved {split}: {images_tensor.shape}, Size: {images_tensor.element_size() * images_tensor.nelement() / (1024**2):.2f} MB")



if __name__ == "__main__":
    input_image_dir = '/Users/ayushbhauwala/Documents/Columbia/Sem 1/DL for CV/project/experimenting/levir_cc/Levir-CC-dataset/images'
    save_dir = './data/LEVIR_CC/processed_images/'
    captions_file = '/Users/ayushbhauwala/Documents/Columbia/Sem 1/DL for CV/project/experimenting/levir_cc/Levir-CC-dataset/LevirCCcaptions.json'
    process_and_save_images(input_image_dir, save_dir, captions_file)