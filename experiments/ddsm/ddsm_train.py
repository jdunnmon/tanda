import json
import sys, os
import tensorflow as tf

from dataset import load_ddsm_data
from experiments.train_scripts import flags, train
from experiments.tfs.image import *
from experiments.utils import get_log_dir_path
from functools import partial 
from itertools import chain

#####################################################################
# Additional TF input flags
flags.DEFINE_integer("tfs", 1, "TF set to use")
flags.DEFINE_string("data_dir", "/home/jdunnmon/research/re/projects/tanda/data/ddsm/benign_malignant", \
    "path to test set")
flags.DEFINE_string("label_json", "/home/jdunnmon/research/re/projects/tanda/data/ddsm/benign_malignant/mass_to_label.json", \
    "path to map of imgs to labels")
flags.DEFINE_boolean("validation_set", True, 
    "If False, use validation set as part of training set")

FLAGS = flags.FLAGS
#####################################################################
#####################################################################
# Transformation functions

def TF_null(img, param):
    """For testing"""
    return img


tf_sets = []

tf_sets = []

# Conservative set of TFs also used for natural images (e.g. CIFAR10)
tfs_1 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1]],
    [partial(TF_blur, sigma=0.1)],
    [partial(TF_enhance_contrast, p=p) for p in [1.25, 0.5]],
    [partial(TF_enhance_brightness, p=p) for p in [1.25, 0.5]], 
    [partial(TF_swirl, strength=p) for p in [0.1, -0.1]]
]
tf_sets.append(list(chain.from_iterable(tfs_1)))


# More extensive set of TFs
tfs_2 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5, 15, -15]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1, 0.75, 1.25]],
    [partial(TF_blur, sigma=p) for p in [0.1, 0.25]],
    [partial(TF_shear, shear=p) for p in [0.1, -0.1, 0.25, -0.25]],
    [partial(TF_swirl, strength=p) for p in [0.1, -0.1, 0.25, -0.25]],
    [partial(TF_jitter, ps=p) for p in [(-4,-4), (-3,1), (2,2), (3,-2)]], 
    [partial(TF_enhance_contrast, p=p) for p in [0.75, 1.25, 0.5, 1.5]],
    [partial(TF_enhance_brightness, p=p) for p in [0.75, 1.25, 0.5, 1.5]], 
    [partial(TF_enhance_sharpness, p=p) for p in [0.75, 1.25, 0.5, 1.5]]
]
tf_sets.append(list(chain.from_iterable(tfs_2)))

###TESTING TFS## 

#Test 1: recreate ML4H results with terrible TFs 
tfs_3 = [partial(TF_power, pow_std=p) for p in [0.15, 0.20, 0.22, 0.25]]
tfs_4 = [partial(TF_shear, shear=p) for p in [0.25, 0.5, -0.25, -0.5]]
tfs_5 = [partial(TF_noise, magnitude=1.0, mean=p) for p in [0.0, 1.0, 1.5, -1.0, -1.5]]

tf_sets.append(tfs_3)
tf_sets.append(tfs_4)
tf_sets.append(tfs_5)

#Test 2: try to recreate gains with single TFs 
tfs_6 = [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]]
tfs_7 = [partial(TF_jitter, ps=p) for p in [(-3,1), (2,2), (3,-2)]]
tfs_8 = [partial(TF_blur, sigma=p) for p in [0.1, 0.25]]

tf_sets.append(tfs_6)
tf_sets.append(tfs_7)
tf_sets.append(tfs_8)


tfs_9 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1]],
    [partial(TF_blur, sigma=0.1)],
    [partial(TF_enhance_contrast, p=p) for p in [1.15, 1.05]],
    [partial(TF_swirl, strength=p) for p in [0.1, -0.1]]
]
tf_sets.append(list(chain.from_iterable(tfs_9)))

tfs_10 = [TF_crop_pad_flip]
tf_sets.append(tfs_10)

tfs_11 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1]],
    [partial(TF_enhance_contrast, p=p) for p in [1.15, 1.05]],
    [TF_mammo_brightness]
]
tf_sets.append(list(chain.from_iterable(tfs_11)))

tfs_12 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1]],
    [partial(TF_enhance_contrast, p=p) for p in [1.15, 1.05]],
    [partial(TF_blur, sigma=0.1)],
    [partial(TF_translate_structure_with_tissue, translation=p, dim=100) \
        for p in [(10,10), (-10,-10), (5,10), (-10,5)]],
    [partial(TF_translate_structure_with_tissue, dim=100)], 
    [partial(TF_rotate_structure_with_tissue, p=p, dim=100) \
        for p in [2.5, -2.5, 15, -15]]
]
tf_sets.append(list(chain.from_iterable(tfs_12)))

tfs_13 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.98, 1.02]],
    [partial(TF_enhance_contrast, p=p) for p in [0.95, 1.05]],
    [partial(TF_translate_structure_with_tissue, dim=100)],
    [partial(TF_rotate_structure_with_tissue, dim=100)]
]
tf_sets.append(list(chain.from_iterable(tfs_13)))

tfs_14 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.98, 1.02]],
    [partial(TF_enhance_contrast, p=p) for p in [0.95, 1.05]],
    [partial(TF_translate_structure_with_tissue, translation=p, dim=100) \
          for p in [(0,3), (0,-3), (3,0), (-3,0), (0,0)]],
    [partial(TF_rotate_structure_with_tissue, p=p, dim=100) \
        for p in [2.5, -2.5, 5, -5]]
]
tf_sets.append(list(chain.from_iterable(tfs_14)))

tfs_15 = [partial(TF_zoom, scale=p) for p in [0.9, 1.1, 0.75, 1.25]]
tfs_16 = [partial(TF_enhance_contrast, p=p) for p in [1.15, 1.05]]
tfs_17 = [TF_mammo_brightness]
tfs_18 = [partial(TF_enhance_sharpness, p=p) for p in [0.75, 1.25, 0.5, 1.5]]
tf_sets.append(tfs_15)
tf_sets.append(tfs_16)
tf_sets.append(tfs_17)
tf_sets.append(tfs_18)


tfs_19 = [partial(TF_translate_structure_with_tissue, translation=p, dim=100) \
            for p in [(0,3), (0,-3), (3,0), (-3,0), (0,0)]]
tfs_20 = [partial(TF_rotate_structure_with_tissue, p=p, dim=100) \
            for p in [2.5, -2.5, 5, -5]]
tfs_21 = [TF_translate_structure_with_tissue]
tfs_22 = [TF_rotate_structure_with_tissue]
tf_sets.append(tfs_19)
tf_sets.append(tfs_20)
tf_sets.append(tfs_21)
tf_sets.append(tfs_22)

tfs_23 = [
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1]],
    [partial(TF_enhance_contrast, p=p) for p in [1.15, 1.05]],
]
tf_sets.append(list(chain.from_iterable(tfs_23)))



#Test 2: Add randomness to TFs? 
tfs = tf_sets[FLAGS.tfs - 1]

#####################################################################

if __name__ == '__main__':

    # Load DDSM mammo data
    dims     = [100, 100, 2]
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_ddsm_data(data_dir=FLAGS.data_dir, \
        label_json=FLAGS.label_json, validation_set=FLAGS.validation_set, segmentations=True)

    print "X_train.shape:", X_train.shape
    # Run training scripts
    train(X_train, dims, tfs, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
        n_classes=2)
