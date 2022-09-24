from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
from random import randrange


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.hr_path = Util.get_paths_from_images(
            '{}/hr_{}'.format(dataroot, r_resolution))
            
        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None


        img_HR = Image.open(self.hr_path[index]).convert("RGB")
      
        crop_size = 256
        orig_size = 500
        x1 = randrange(0, orig_size - crop_size)
        y1 = randrange(0, orig_size - crop_size)
        img_HR = img_HR.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        
        img_LR = img_HR.resize((64, 64), resample=Image.BICUBIC)
        img_LR = img_LR.resize((crop_size, crop_size), resample=Image.BICUBIC)
        img_SR = img_LR
        
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
