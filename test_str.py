import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

import cv2

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print("Failed to import pydensecrf,\
           CRF post-processing will not work")

ASPECT_AWARE_SCALING = True
TARGET_SIZE = 256
MAX_SIZE = 2048

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')] + '_two_heads'

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    # img = misc.imread(args.img_path)
    img = cv2.imread(args.img_path)

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes
    
    if ASPECT_AWARE_SCALING:
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(TARGET_SIZE) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > MAX_SIZE:
            im_scale = float(MAX_SIZE) / float(im_size_max)
        resized_img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
    else:
        resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp='bicubic')

    orig_size = img.shape[:-1]
    if model_name in ['pspnet', 'icnet', 'icnetBN']:
        if ASPECT_AWARE_SCALING:
            # Make sure size attributes are even numbers
            out_shape = im_shape[0:2] * im_scale 
            out_shape = out_shape//2*2+1
            img = cv2.resize(img, out_shape, None, interpolation=cv2.INTER_LINEAR)
        else:
            img = misc.imresize(img, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
    else:
        if ASPECT_AWARE_SCALING:
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        else:
            img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

    # img = img[:, :, ::-1] # RGB -> BRG
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        if torch.cuda.is_available():
            model.cuda(0)
            images = Variable(img.cuda(0))
        else:
            images = Variable(img)

        output_first_head, output_second_head = model(images)
        #outputs = F.softmax(outputs, dim=1)

    if args.dcrf:
        for idx, out in enumerate([output_first_head, output_second_head]):
            unary = out.data.cpu().numpy()
            unary = np.squeeze(unary, 0)
            unary = -np.log(unary)
            unary = unary.transpose(2, 1, 0)
            w, h, c = unary.shape
            unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
            unary = np.ascontiguousarray(unary)
           
            resized_img = np.ascontiguousarray(resized_img)

            d = dcrf.DenseCRF2D(w, h, loader.n_classes)
            d.setUnaryEnergy(unary)
            d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

            q = d.inference(50)
            mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
            decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
            dcrf_path = args.out_path + 'out_drf_' + str(idx) + '.png'
            misc.imsave(dcrf_path, decoded_crf)
            print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred_first_head = np.squeeze(output_first_head.data.max(1)[1].cpu().numpy(), axis=0)
    pred_second_head = np.squeeze(output_second_head.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ['pspnet', 'icnet', 'icnetBN']:
        pred_first_head = pred_first_head.astype(np.float32)
        pred_first_head = misc.imresize(pred_first_head, orig_size, 'nearest', mode='F') # float32 with F mode, resize back to orig_size
        pred_second_head = pred_second_head.astype(np.float32)
        pred_second_head = misc.imresize(pred_second_head, orig_size, 'nearest', mode='F') # float32 with F mode, resize back to orig_size

    decoded_first_head = loader.decode_segmap(pred_first_head)
    decoded_second_head = loader.decode_segmap(pred_second_head)
    print('Classes found (first head):', np.unique(pred_first_head), ' | Classes found (second head):', np.unique(pred_second_head))
    misc.imsave(args.out_path + 'out_0.png', decoded_first_head)
    misc.imsave(args.out_path + 'out_1.png', decoded_second_head)
    print("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | False by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--dcrf', dest='dcrf', action='store_true', 
                        help='Enable DenseCRF based post-processing | False by default')
    parser.add_argument('--no-dcrf', dest='dcrf', action='store_false', 
                        help='Disable DenseCRF based post-processing | False by default')
    parser.set_defaults(dcrf=False)

    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)
