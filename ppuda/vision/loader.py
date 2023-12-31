# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Prepares training and evaluation torch DataLoaders.
Supports ImageNet and torchvision datasets, such as CIFAR-10 and CIFAR-100.

"""


import os
import torch
from torchvision.datasets import *
from torch.utils.data import DataLoader
#ADDED CODE
from .transforms import transforms_cifar, transforms_imagenet, transforms_caltech
from .imagenet import ImageNetDataset
#ADDED CODE
from .caltech import Caltech101Dataset
import torchvision.datasets as datasets
from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

def image_loader(dataset='imagenet', data_dir='./data/', test=True, im_size=32,
                 batch_size=64, test_batch_size=64, num_workers=0,
                 cutout=False, cutout_length=16, noise=False,
                 seed=1111, load_train_anyway=False, n_shots=None):
    """

    :param dataset: image dataset: imagenet, cifar10, cifar100, etc.
    :param data_dir: location of the dataset
    :param test: True to load the test data for evaluation, False to load the validation data.
    :param im_size: image size for CIFAR data (ignored for dataset='imagenet')
    :param batch_size: training batch size of images
    :param test_batch_size: evaluation batch size of images
    :param num_workers: number of threads to load/preprocess images
    :param cutout: use Cutout for data augmentation from
                   "Terrance DeVries, Graham W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017."
    :param cutout_length: Cutout hyperparameter
    :param noise: evaluate on the images with added Gaussian noise
    :param seed: random seed to shuffle validation images on ImageNet
    :param load_train_anyway: load training images even when evaluating on test data (test=True)
    :param n_shots: the number of training images per class (only for CIFAR-10 and other torchvision datasets and when test=True)
    :return: training and evaluation torch DataLoaders and number of classes in the dataset
    """
    train_data = None

    if dataset.lower() == 'imagenet':
        train_transform, valid_transform = transforms_imagenet(noise=noise, cifar_style=False)
        imagenet_dir = os.path.join(data_dir, 'imagenet')

        if not test or load_train_anyway:
            train_data = ImageNetDataset(imagenet_dir, 'train', transform=train_transform, has_validation=not test)

        valid_data = ImageNetDataset(imagenet_dir, 'val', transform=valid_transform, has_validation=test)
        print(type(valid_data))
        shuffle_val = True  # to evaluate models with batch norm in the training mode (in case there is no running statistics)
        n_classes = 101
        generator = torch.Generator()
        generator.manual_seed(seed)  # to reproduce evaluation with shuffle=True on ImageNet

#ADDED CODE
    if dataset.lower() == 'caltech':
        print("Loading caltech categories and annotations..")
        train_transform, valid_transform = transforms_caltech(noise=noise, cifar_style=False)
        caltech_dir = os.path.join(data_dir,  'caltech-101')

        image_paths = list(paths.list_images(caltech_dir + '/101_ObjectCategories'))
        
        data = []
        labels = []
        label_names = []
        for image_path in image_paths:
            label = image_path.split(os.path.sep)[-2]
            if label == 'BACKGROUND_Google':
                continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data.append(image)
            label_names.append(label)
            labels.append(label)

        data = np.array(data)
        labels = np.array(labels)

        label_map = {}

        # Create the mapping
        for label in labels:
            if label not in label_map:
                label_map[label] = len(label_map)

        print(label_map)
        # Use the mapping to convert the labels to integers
        labels_int = [label_map[label] for label in labels]
        labels = labels_int

        (x_train, x_val, y_train, y_val) = train_test_split(data, labels, 
                                                            test_size=0.25, 
                                                            random_state=42)
        
        valid_data = Caltech101Dataset("val", x_val, y_val,  transforms=valid_transform)
        valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)
        print("Val loaded..")

        if not test or load_train_anyway:
            train_data = Caltech101Dataset("train", x_train, y_train, transforms=train_transform)
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            print("train data loaded..")
        else:
            train_loader = None

        shuffle_val = True  # to evaluate models with batch norm in the training mode (in case there is no running statistics)
        n_classes = 101
        generator = torch.Generator()
        generator.manual_seed(seed)  # to reproduce evaluation with shuffle=True on ImageNet

        return train_loader, valid_loader, n_classes
    
    # if dataset.lower() == 'caltech':
    #     train_transform, valid_transform = transforms_caltech(noise=noise, cifar_style=False)
    #     caltech_dir = os.path.join(data_dir, 'caltech-101')

    #     # Load the dataset
    #     data = datasets.Caltech101(caltech_dir, download=True)

    #     # Define the number of samples to be in the training and validation sets
    #     train_size = int(0.8 * len(data))
    #     val_size = len(data) - train_size
    #     weights = [1.0 / 101] * 101 

    #     train_data = torch.utils.data.sampler.WeightedRandomSampler(weights, train_size)

    #     if not test or load_train_anyway:
    #         train_data = torch.utils.data.sampler.WeightedRandomSampler(weights, train_size)
    #         train_loader = DataLoader(train_data, batch_size=32, sampler=train_data, num_workers=4)
    #         train_loader.dataset.transform = train_transform

    #     valid_data = torch.utils.data.sampler.WeightedRandomSampler(weights, val_size)
    #     val_loader = DataLoader(valid_data, batch_size=32, shuffle=True, num_workers=4)
    #     val_loader.dataset.transform = valid_transform

    #     print(type(valid_data))
    #     shuffle_val = True  # to evaluate models with batch norm in the training mode (in case there is no running statistics)
    #     n_classes = len(valid_data.classes)
    #     generator = torch.Generator()
    #     generator.manual_seed(seed)  # to reproduce evaluation with shuffle=True on ImageNet 

    #     return train_loader, valid_loader, n_classes 

    else:
        # if dataset == 'Caltech101':
        #     train_transform, valid_transform = transforms_caltech(noise=noise, cifar_style=False)
        # else:
        dataset = dataset.upper()
        train_transform, valid_transform = transforms_cifar(cutout=cutout, cutout_length=cutout_length, noise=noise, sz=im_size)
        
        if test:
            valid_data = eval('{}(data_dir, train=False, download=True, transform=valid_transform)'.format(dataset))
            if load_train_anyway:
                train_data = eval('{}(data_dir, train=True, download=True, transform=train_transform)'.format(dataset))
                if n_shots is not None:
                    train_data = to_few_shot(train_data, n_shots=n_shots)
        else:
            if n_shots is not None:
                print('few shot regime is only supported for evaluation on the test data')
            # Held out 10% (e.g. 5000 images in case of CIFAR-10) of training data as the validation set
            train_data = eval('{}(data_dir, train=True, download=True, transform=train_transform)'.format(dataset))
            valid_data = eval('{}(data_dir, train=True, download=True, transform=valid_transform)'.format(dataset))
            n_all = len(train_data.targets)
            n_val = n_all // 10
            idx_train, idx_val = torch.split(torch.arange(n_all), [n_all - n_val, n_val])

            train_data.data = train_data.data[idx_train]
            train_data.targets = [train_data.targets[i] for i in idx_train]

            valid_data.data = valid_data.data[idx_val]
            valid_data.targets = [valid_data.targets[i] for i in idx_val]

            if n_shots is not None:
                train_data = to_few_shot(train_data, n_shots=n_shots)

        if train_data is not None:
            train_data.checksum = train_data.data.mean()
            train_data.num_examples = len(train_data.targets)
            print(type(train_data))
        shuffle_val = False
        n_classes = len(torch.unique(torch.tensor(valid_data.targets)))
        generator = None

        valid_data.checksum = valid_data.data.mean()
        valid_data.num_examples = len(valid_data.targets)

    print('loaded {}: {} classes, {} train samples (checksum={}), '
          '{} {} samples (checksum={:.3f})'.format(dataset,
                                                   n_classes,
                                                   train_data.num_examples if train_data else 'none',
                                                   ('%.3f' % train_data.checksum) if train_data else 'none',
                                                   valid_data.num_examples,
                                                   'test' if test else 'val',
                                                   valid_data.checksum))


    if train_data is None:
        train_loader = None
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)
        # if dataset.lower() == 'caltech':
        #     train_data.dataset.transform = train_transform

    valid_loader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=shuffle_val,
                              pin_memory=True, num_workers=num_workers, generator=generator)

    # if dataset.lower() == 'caltech':
    #     valid_data.dataset.transform = valid_transform

    return train_loader, valid_loader, n_classes
##TLL HERE

def to_few_shot(dataset, n_shots=100):
    """
    Transforms torchvision dataset to a few-shot dataset.
    :param dataset: torchvision dataset
    :param n_shots: number of samples per class
    :return: few-shot torchvision dataset
    """
    try:
        targets = dataset.targets  # targets or labels depending on the dataset
        is_targets = True
    except:
        targets = dataset.labels
        is_targets = False

    assert min(targets) == 0, 'labels should start from 0, not from {}'.format(min(targets))

    # Find n_shots samples for each class
    labels_dict = {}
    for i, lbl in enumerate(targets):
        lbl = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
        if lbl not in labels_dict:
            labels_dict[lbl] = []
        if len(labels_dict[lbl]) < n_shots:
            labels_dict[lbl].append(i)

    idx = sorted(torch.cat([torch.tensor(v) for k, v in labels_dict.items()]))  # sort according to the original order in the full dataset

    dataset.data = [dataset.data[i] for i in idx] if isinstance(dataset.data, list) else dataset.data[idx]
    targets = [targets[i] for i in idx]
    if is_targets:
        dataset.targets = targets
    else:
        dataset.labels = targets

    return dataset
