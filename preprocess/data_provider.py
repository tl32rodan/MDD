import numpy as np
from .data_list import ImageList
import torch.utils.data as util_data
from torchvision import transforms
from PIL import Image, ImageOps


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def load_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224, is_cen=False, root_folder='./data'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if not is_train:
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        images = ImageList(open(images_file_path).readlines(), transform=transformer, root_folder=root_folder)
        #images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        if is_cen:
            transformer = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize])
        else:
            transformer = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        images = ImageList(open(images_file_path).readlines(), transform=transformer, root_folder=root_folder)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4)
    return images_loader

