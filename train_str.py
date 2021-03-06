import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *

def train(args):

    # Setup Augmentations
    # data_aug= Compose([RandomRotate(10), RandomHorizontallyFlip()])
    data_aug= Compose([RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)
    # t_loader = data_loader(data_path, is_transform=True, img_size=None, augmentations=data_aug, img_norm=args.img_norm)
    # v_loader = data_loader(data_path, is_transform=True, split='val', img_size=None, img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics_first_head = runningScore(n_classes)
    running_metrics_second_head = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou_first_head = -100.0 
    best_iou_second_head = -100.0 
    class_weights = torch.ones(n_classes).cuda()
    class_weights[-1] *= 5.0 # Distinguishing the border is the most important task
    print ("Class weights:", class_weights)
    for epoch in range(args.n_epoch):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            output_first_head, output_second_head = model(images)

            loss_row = loss_fn(input=output_first_head, target=labels[:, 0, :, :], weight=class_weights)
            loss_col = loss_fn(input=output_second_head, target=labels[:, 1, :, :], weight=class_weights)
            loss = loss_row + loss_col

            loss.backward()
            optimizer.step()

            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.item()]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.item()))

        model.eval()
        avg_loss_val = 0.0
        num_iter = 0
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())

                # outputs = model(images_val)
                output_first_head, output_second_head = model(images_val)
                pred_first_head = output_first_head.data.max(1)[1].cpu().numpy()
                pred_second_head = output_second_head.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                gt_first_head = gt[:, 0, :, :]
                gt_second_head = gt[:, 1, :, :]
                running_metrics_first_head.update(gt_first_head, pred_first_head)
                running_metrics_second_head.update(gt_second_head, pred_second_head)

                # Compute loss on the validation set
                loss_row = loss_fn(input=output_first_head, target=labels_val[:, 0, :, :], weight=class_weights)
                loss_col = loss_fn(input=output_second_head, target=labels_val[:, 1, :, :], weight=class_weights)
                loss_val = loss_row + loss_col

                avg_loss_val += loss_val.item()
                num_iter += 1
        
        # Update the learning rate
        avg_loss_val = avg_loss_val / num_iter
        print ("Average validation loss: %.4f" % (avg_loss_val))
        scheduler.step(avg_loss_val)

        score_first_head, class_iou_first_head = running_metrics_first_head.get_scores()
        score_second_head, class_iou_second_head = running_metrics_second_head.get_scores()
        print ("First head:")
        for k, v in score_first_head.items():
            print(k, v)
        print ("Second head:")
        for k, v in score_second_head.items():
            print(k, v)
        running_metrics_first_head.reset()
        running_metrics_second_head.reset()

        if score_first_head['Mean IoU : \t'] >= best_iou_first_head and score_second_head['Mean IoU : \t'] >= best_iou_second_head:
            best_iou_first_head = score_first_head['Mean IoU : \t']
            best_iou_second_head = score_second_head['Mean IoU : \t']
            print ("Saving best model with %.5f first head mean IoU and %.5f second head mean IoU" % (best_iou_first_head, best_iou_second_head))
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}_{}_best_model.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)
