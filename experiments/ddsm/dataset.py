
import numpy as np
import cPickle
import os
import shutil
import json

from skimage import img_as_float
from functools import partial
from PIL import Image

from experiments.tfs.image import *

IMAGE_SIZE = 100

def to_one_hot(y, n_classes=2):
    Y = np.zeros([y.shape[0], n_classes])
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y

def load_files(dirs, label_json, segmentations):     
    train_dir, val_dir, test_dir, masks_dir = dirs
    train_imgs = os.listdir(train_dir)
    val_imgs = os.listdir(val_dir)
    test_imgs = os.listdir(test_dir)
    mask_imgs = os.listdir(masks_dir)

    ##open json file 
    with open(label_json, 'r') as fp: 
        img_to_label = json.load(fp)
    assert type(img_to_label.values()[0]) == int, 'labels are not ints.'

    def append_lists(img_dir, imgs, mask_dir=None, mask_imgs=None): 
        X, Y, masks = [], [], []
        for i,img_file in enumerate(imgs): 
            X.append(os.path.join(img_dir, img_file))
            Y.append(img_to_label[img_file])
            if mask_dir: 
                masks.append(os.path.join(mask_dir, mask_imgs[i]))
        if mask_dir: 
            return X, Y, masks

        return X,Y

    masks = None
    if segmentations: 
        train_imgs = [img for img in train_imgs if img in mask_imgs]
        X_train, Y_train, masks = append_lists(train_dir, train_imgs, \
            mask_dir=masks_dir, mask_imgs=mask_imgs)
    else: 
        X_train, Y_train = append_lists(train_dir, train_imgs)

    X_valid, Y_valid = append_lists(val_dir, val_imgs)
    X_test, Y_test = append_lists(test_dir, test_imgs)

    return X_train, Y_train, masks, X_valid, Y_valid, X_test, Y_test

def convert_file(files, y, as_float, masks=None): 
    Y = to_one_hot(np.array(y))

    if not masks: 
        X = np.zeros((len(files), IMAGE_SIZE, IMAGE_SIZE, 1))
    else: 
        X = np.zeros((len(files), IMAGE_SIZE, IMAGE_SIZE, 2))

    print 'loading in %d images...' % len(files)
    for i, example in enumerate(files): 
        im = Image.open(example)
        im = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)))
        im = im[:,:,np.newaxis]

        if as_float:
            X[i,:,:,0] = np.squeeze(img_as_float(im))

        if masks: 

            if len(masks) != 1:
                mask = Image.open(masks[i])
                mask = np.array(mask.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
                mask = mask[:,:,np.newaxis]
                X[i,:,:,1] = np.squeeze(img_as_float(mask))
            else: 
                X[i,:,:,1] = img_as_float(np.zeros((IMAGE_SIZE, IMAGE_SIZE)))

    return X, Y 

def load_ddsm_data(data_dir, label_json, \
    validation_set=True, segmentations=True, \
    one_hot=True, as_float=True):
    """Read in train, validation, test files"""
    train_dir = os.path.join(data_dir, 'train_set')
    val_dir = os.path.join(data_dir, 'val_set')
    test_dir = os.path.join(data_dir, 'test_set')
    masks_dir = os.path.join(data_dir, 'masks_cropped')
    dirs = [train_dir, val_dir, test_dir, masks_dir]
    X_tr_files, y_tr, masks, X_val_files, y_val, X_test_files, y_test = \
        load_files(dirs, label_json, segmentations)

    if not validation_set: 
        X_tr_files += X_val_files 
        y_tr += y_val 

    """Load in train, validation, test images""" 
    X_train, Y_train = convert_file(X_tr_files, y_tr, as_float, masks=masks)

    if segmentations: 
        X_valid, Y_valid = convert_file(X_val_files, y_val, as_float, masks=[0])
        X_test, Y_test = convert_file(X_test_files, y_test, as_float, masks=[0])
    else: 
        X_valid, Y_valid = convert_file(X_val_files, y_val, as_float)
        X_test, Y_test = convert_file(X_test_files, y_test, as_float)

    if not validation_set: 
        X_valid, Y_valid = X_test, Y_test

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

if __name__ == '__main__':
    data_dir = './data/benign_malignant'
    label_json = './data/json/mass_to_label.json'
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
        load_ddsm_data(data_dir, label_json, segmentations=True)
    print X_train.shape
    print Y_train.shape
    print X_valid.shape
    print Y_valid.shape
    print X_test.shape
    print Y_test.shape

    # data_root = './data'
    # train_dir = os.path.join(data_root, 'train_set')
    # train_files = [os.path.join(data_root + '/train_set', file) \
    #     for file in os.listdir(train_dir)]
    # debug_files = np.random.choice(train_files, size=6)

    # params = np.linspace(0.001,0.2,num=10)
    # TF = TF_noise
    # for file in debug_files: 
    #     shutil.copy(file, './test_imgs')
    #     for p in params: 
    #         print 'file:', file
    #         im = Image.open(file)
    #         im = np.array(im, dtype=np.uint8)
    #         im = im[:,:,np.newaxis]
    #         im = img_as_float(im)
    #         im = TF(im, magnitude=0.01, mean=p)
    #         file_name = file.split('/')[-1]
    #         np_to_pil(im).save('./test_imgs/test_image_noise_' + file_name[:-4] + str(p) + '.tif')

