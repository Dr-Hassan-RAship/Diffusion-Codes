import os
import pickle
import logging
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# [CHANGED] imported random
import random

# [CHANGED] --> Added a function which takes the image and masks directory and generates a pickel file containing the file names
# train and test split (90 - 10)

def generate_pickel_file(root_dir, split = 0.9):
    # Get img and mask files
    img_dir  = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')
    
    # Get list of image and mask files
    img_files  = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    # Get a random batch of indices of img_dir (for train) and its corresponding mask_file indidces and then the 
    # remaining goes to test. 
    
    # In the end a .pkl object of a dict should be saved in the root_dir which should contain two keys 'train' and 'test'
    # where each key a dict value. The dict value should have key 'name_list' and its value should be a list of the file names
    # chosen for the train/test
    
    # E.g 
    # {'train': {'name_list': [file_name1_train.png, file_name2_train.png]}, 'test': {'name_list': [file_name1_test.png, file_name2_test.png]}}
    
    # Extract base filenames (without extension) to match pairs
    img_bases  = [os.path.splitext(f)[0] for f in img_files]
    mask_bases = [os.path.splitext(f)[0] for f in mask_files]
    
    # Find common base filenames (ensure paired image-mask)
    common_bases = sorted(list(set(img_bases) & set(mask_bases)))
    
    # Shuffle for random split
    random.seed(42)  # For reproducibility
    random.shuffle(common_bases)
    
    # Split into train and test (90-10)
    train_size  = int(split * len(common_bases))
    train_bases = common_bases[:train_size]
    test_bases  = common_bases[train_size:]
    
    # Create dictionary for pickle file
    pickle_dict = {
        'train': {'name_list': train_bases},
        'test': {'name_list': test_bases}
    }
    
    # Save pickle file in root_dir
    pickle_path = os.path.join(root_dir, 'kvasir-seg_train_test_names.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_dict, f)
    
    print(f"Pickle file saved at {pickle_path}")
    print(f"Train set: {len(train_bases)} files, Test set: {len(test_bases)} files")
    
    return pickle_path
    
class Image_Dataset(Dataset):
    def __init__(self, pickle_file_path, stage='train') -> None:
        super().__init__()
        with open(pickle_file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
        self.img_path          = os.path.join(os.path.dirname(pickle_file_path), 'images')
        self.mask_path         = os.path.join(os.path.dirname(pickle_file_path), 'masks')
        self.img_size          = 224
        self.stage             = stage
        self.name_list         = loaded_dict[stage]['name_list']
        self.transform         = self.get_transforms()
        logging.info('{} set num: {}'.format(stage, len(self.name_list)))

        del loaded_dict
    
    # [CHANGED] --> Removed always_apply=True as it is giving error or warning for ToFloat, Resize, ToTensorV2
    def get_transforms(self):
        if self.stage == 'train':
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.2),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=20, p=0.4),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        else:
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        return transforms

    def __getitem__(self, index):
        name = self.name_list[index]
        # load img & seg
        # [CHANGED] --> from .png to jpg as Kvasir-SEG has .png files. Also note both image and mask are being loaded as RGB
        seg_image = Image.open(os.path.join(self.mask_path, name + '.png')).convert("RGB")
        seg_data  = np.array(seg_image).astype(np.float32)
        img_image = Image.open(os.path.join(self.img_path,  name + '.png')).convert("RGB")
        img_data  = np.array(img_image).astype(np.float32)

        augmented = self.transform(image=img_data, mask=seg_data)

        aug_img = augmented['image']
        aug_seg = augmented['mask']

        return {
            'name': name,
            'img': aug_img,
            'seg': aug_seg
        }

    def __len__(self):
        return len(self.name_list)

