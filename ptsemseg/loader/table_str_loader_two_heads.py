import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from torch.utils import data
import cv2

ASPECT_AWARE_SCALING = True

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

class TableStrLoaderTwoHeads(data.Dataset):
    """Data loader for the Table Structure Recognition dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=None, augmentations=None, img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 3
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (None if img_size is None else (img_size, img_size))
        for split in ['train', 'val']:
            path = pjoin(self.root, 'ImageSets', split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, 'Images',  im_name + '.jpg')
        lbl_path_row = pjoin(self.root, 'pre_encoded', im_name + '-row.png')
        lbl_path_col = pjoin(self.root, 'pre_encoded', im_name + '-column.png')
        im = cv2.imread(im_path)
        im = np.array(im, dtype=np.uint8)
        lbl_row = cv2.imread(lbl_path_row, cv2.IMREAD_GRAYSCALE)
        lbl_row = np.array(lbl_row, dtype=np.uint8)
        lbl_col = m.imread(lbl_path_col, cv2.IMREAD_GRAYSCALE)
        lbl_col = np.array(lbl_col, dtype=np.uint8)
        lbl = np.zeros((lbl_row.shape[0], lbl_row.shape[1]), dtype=lbl_row.dtype)
        lbl = np.stack((lbl_row, lbl_col, lbl), axis=2)
        # print ("Before:", im.shape, lbl.shape)

        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        lbl = lbl[:-1, :, :] # Discard the 3rd dimension
        # print ("Data shape:", im.size(), lbl.size())
        return im, lbl


    def transform(self, img, lbl):
        if self.img_size is not None:
            if ASPECT_AWARE_SCALING:
                target_size = self.img_size[0]
                max_size = self.img_size[1]
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(target_size) / float(im_size_min)
                # Prevent the biggest axis from being more than MAX_SIZE
                if np.round(im_scale * im_size_max) > max_size:
                    im_scale = float(max_size) / float(im_size_max)
                img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            else:
                # img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
                img = cv2.resize(img, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST) # uint8 with RGB mode
        # img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        # print ("Before:", np.unique(lbl))
        if self.img_size is not None:
            if ASPECT_AWARE_SCALING:
                lbl = cv2.resize(lbl, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
            else:
                # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', 'F')
                lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST) # uint8 with RGB mode
        lbl = lbl.astype(int)
        # NHWC -> NCHW
        lbl = lbl.transpose(2, 0, 1)
        # print (np.unique(lbl))
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    def get_table_str_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        # CLASS_MASK = {(0, 0, 0): 0, (0, 128, 0): 128, (0, 0, 128): 128, (192, 224, 224): 255}
        colors = np.asarray([[0,0,0], [0,128,0], [128,0,0], [224,224,192]])
        labels = np.asarray([0, 1, 1, 2]) # Give even high weight to the boundary
        # labels = np.asarray([0, 128, 128, 255])
        return zip(colors, labels)


    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        # print (np.unique(mask.reshape(-1, mask.shape[2]), axis=0)) # Print unique pixel coords
        for color, label in self.get_table_str_labels():
            label_mask[np.where(np.all(mask == color, axis=-1))] = label
        label_mask = label_mask.astype(int)
        # print (np.unique(label_mask))
        return label_mask


    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_table_str_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        # for ll in range(0, self.n_classes):
        for color, label in label_colours:
            b[label_mask == label] = color[0]
            g[label_mask == label] = color[1]
            r[label_mask == label] = color[2]
            # b[label_mask == ll] = label_colours[ll, 2]

        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = get_data_path('table_str')
        target_path = pjoin(self.root, 'pre_encoded')
        if not os.path.exists(target_path): os.makedirs(target_path) 
        # path = pjoin(sbd_path, 'dataset/train.txt')
        # sbd_train_list = tuple(open(path, 'r'))
        # sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        # train_aug = self.files['train'] + sbd_train_list

        # # keep unique elements (stable)
        # train_aug = [train_aug[i] for i in \
        #                   sorted(np.unique(train_aug, return_index=True)[1])]
        # self.files['train_aug'] = train_aug
        # set_diff = set(self.files['val']) - set(train_aug) # remove overlap
        # self.files['train_aug_val'] = list(set_diff)

        pre_encoded = glob.glob(pjoin(target_path, '*.png'))
        expected = np.unique(self.files['train'] + self.files['val']).size * 2
        # expected = np.unique(self.files['train_aug'] + self.files['val']).size

        if len(pre_encoded) != expected:
            print ("Expected number of files:", expected, " | Found number of files:", len(pre_encoded))
            print("Pre-encoding segmentation masks...")
            # for ii in tqdm(sbd_train_list):
            #     lbl_path = pjoin(sbd_path, 'dataset/cls', ii + '.mat')
            #     data = io.loadmat(lbl_path)
            #     lbl = data['GTcls'][0]['Segmentation'][0].astype(np.int32)
            #     lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
            #     m.imsave(pjoin(target_path, ii + '.png'), lbl)

            for ii in tqdm(self.files['train'] + self.files['val']):
                # fname = ii + '.png'
                fname_row = ii + '-row.png'
                fname_col = ii + '-column.png'
                lbl_path_row = pjoin(self.root, 'SegmentationClass', fname_row)
                lbl_path_col = pjoin(self.root, 'SegmentationClass', fname_col)
                lbl_row = self.encode_segmap(m.imread(lbl_path_row))
                lbl_row = m.toimage(lbl_row, high=lbl_row.max(), low=lbl_row.min())
                lbl_col = self.encode_segmap(m.imread(lbl_path_col))
                lbl_col = m.toimage(lbl_col, high=lbl_col.max(), low=lbl_col.min())
                # m.imsave(pjoin(target_path, fname), lbl)
                m.imsave(pjoin(target_path, fname_row), lbl_row)
                m.imsave(pjoin(target_path, fname_col), lbl_col)

        # assert expected == 9733, 'unexpected dataset sizes'

# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
# if __name__ == '__main__':
#     local_path = '/netscratch/siddiqui/TableDetection/icdar_str_devkit/data/'
#     bs = 4
#     augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
#     dst = pascalVOCLoader(root=local_path, is_transform=True, augmentations=augs)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()[:, ::-1, :, :]
#         imgs = np.transpose(imgs, [0,2,3,1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             axarr[j][0].imshow(imgs[j])
#             axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
