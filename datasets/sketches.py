import os
import glob
from PIL import Image
import torch.utils.data as data
import numpy as np

class Sketches(data.Dataset):
    """Whatnick Ink Sketches <https://www.whatnick.ink> Dataset

    **Parameters:**
        - **root** (string): Root directory of dataset where directories 'processed' and 'raw' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val'
        - **transfrom** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, split='train', transform=None):
        self.root_dir = root
        self.images_folder = os.path.join(root, "unprocessed")
        self.masks_folder = os.path.join(root, "processed")
        self.file_list = list(glob.glob(os.path.join(self.masks_folder,"*.png")))

        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    

    def __getitem__(self, idx):
        sketch_name = os.path.basename(self.file_list[idx])
        image_name = sketch_name.split(".")[0]
        img_name = os.path.join(self.images_folder, f"{image_name}.jpg")
        mask_name = os.path.join(self.masks_folder, f"{image_name}.png")

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name)

        # Transform the mask so that it has binary values (0 or 255)
        # Extract the alpha channel (transparency) from the mask
        alpha_channel = mask.split()[-1]

        # Threshold the alpha channel to create a binary mask
        mask = alpha_channel.point(lambda p: p > 0).convert("L")

        # Define any additional data transformations if needed (e.g., normalization)
        # ...

        return {"image": np.array(image), "mask": np.array(mask)}