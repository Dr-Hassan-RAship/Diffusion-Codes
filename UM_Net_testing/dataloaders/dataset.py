#------------------------------------------------------------------------------#
# File name         : dataset.py
# Purpose           : Preparation of Polyp Dataset Dataloader
#
# Authors           : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email             : hassan.mohyuddin@lums.edu.pk
#
# Last Date         : June 11, 2025
#------------------------------------------------------------------------------#

import os, cv2
import numpy                                                as np

from torchvision                        import transforms
from torch.utils.data                   import Dataset
from dataloaders.transform              import *

from torch.nn                           import functional           as Fun
from torchvision.transforms             import functional           as F

#------------------------------------------------------------------------------#
def get_mean_and_std(img):
    x_mean, x_std   = cv2.meanStdDev(img)
    x_mean          = np.hstack(np.around(x_mean, 2))
    x_std           = np.hstack(np.around(x_std, 2))

    return x_mean, x_std

#------------------------------------------------------------------------------#
class Polyp_Dataset(Dataset):
    def __init__(self, root, mode, slice_size):
        super(Polyp_Dataset, self).__init__()
        self.mode       = mode
        self.slice_size = slice_size
        self.imglist    = []
        self.gtlist     = []
        self.color      = []

        data_path       = os.path.join(root, mode)
        datalist        = os.listdir(data_path + '/images')

        for data in datalist:
            self.imglist.append(os.path.join(data_path + '/images', data))
            self.gtlist.append(os.path.join(data_path + '/masks', data))

        self.color      = self.imglist.copy()
        # transfer_data   = os.listdir(root + 'color_transfer')
        # for name in transfer_data:
        #     self.color.append(os.path.join(root + 'color_transfer', name))


        if mode == 'train':
            transform = transforms.Compose([
                Resize(self.slice_size),  # (288, 384), (320, 320), (352, 352)
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
                RandomRotation(90),
                RandomZoom((0.9, 1.1)),
                #RandomCrop((224, 224)),
                ToTensor(),
                Normalization()])

        elif mode == 'val' or mode == 'test':
            transform = transforms.Compose([
                Resize(self.slice_size),
                ToTensor(),
                Normalization()])

        self.transform  = transform

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        if self.mode == 'train':
            # img_path    = self.imglist[index]
            # gt_path     = self.gtlist[index]
            # file_name   = img_path.split('\\')[-1]
            #
            # img         = Image.open(img_path).convert('RGB')
            # gt          = Image.open(gt_path).convert('L')
            #
            # data        = {'image': img, 'label': gt}
            # data        = self.transform(data)

            # return data
            img_path    = self.imglist[index]
            gt_path     = self.gtlist[index]
            file_name   = img_path.split('\\')[-1]

            img1        = cv2.imread(img_path)
            img1        = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
            img1_mean, img1_std = get_mean_and_std(img1)

            color_path  = self.color[(random.randint(0, len(self.color))) % len(self.color)]
            img2        = cv2.imread(color_path)
            img2        = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
            img2_mean, img2_std = get_mean_and_std(img2)

            img3        = (img1 - img1_mean) / img1_std * img2_std + img2_mean
            np.putmask(img3, img3 > 255, 255)
            np.putmask(img3, img3 < 0, 0)
            image       = cv2.cvtColor(cv2.convertScaleAbs(img3), cv2.COLOR_LAB2RGB)
            image       = Image.fromarray(image)

            gt          = Image.open(gt_path).convert('L')

            data        = {'image': image, 'label': gt}
            data        = self.transform(data)
            #data['file_name'] = file_name

            # print('after transform', data['label'].shape)
            # print(torch.unique(data['label']))

            return data

        elif self.mode == 'val' or self.mode == 'test':
            img_path    = self.imglist[index]
            gt_path     = self.gtlist[index]
            file_name   = img_path.split('\\')[-1]

            img         = Image.open(img_path).convert('RGB')
            gt          = Image.open(gt_path).convert('L')

            data        = {'image': img, 'label': gt}
            data        = self.transform(data)

            data['file_name'] = file_name
            return data

#------------------------------------------------------------------------------#
def utilize_transformation(img, mask, transforms_op):
    """
    Applying transformations to img and mask with same random seed
    """

    state   = torch.get_rng_state()
    mask    = transforms_op(mask)
    torch.set_rng_state(state)
    img     = transforms_op(img)

    return img, mask


# -----------------------------------------------------------------------------#
