import argparse
from functools import partial
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from tanda.discriminator import ResNetDefault, GenericImageCNN
from experiments.utils import balanced_subsample
from experiments.tfs.image import *
from dataset import load_ddsm_data
from experiments.train_scripts import flags, train


CHECKPOINT_ROOTS = [
	'experiments/log/2017_05_18/tan_only_mammo_0517_tfs17_05_59_07/2017_05_18/end_model_tfs17_10per_subsample_23_58_03', # T (0) - deep
	'experiments/log/2017_05_18/end_model_tfs17_10per_subsample_23_59_08', # R - deep
	'experiments/log/2017_05_18/tan_only_mammo_0517_tfs17_05_59_07/2017_05_19/end_model_tfs17_10per_subsample_07_00_10', # T (5) - nebula
]
PATH_SUFFIX = 'end_model/0/checkpoints/model_checkpoint-19'
# CHECKPOINT_PATHS = [os.path.join(r, PATH_SUFFIX) for r in CHECKPOINT_ROOTS]
CHECKPOINT_PATHS = [os.path.join(CHECKPOINT_ROOTS[0], PATH_SUFFIX)]

if __name__ == '__main__':
	data_dir = "/home/zeshanmh/tanda/experiments/ddsm/data/benign_malignant"
	label_json = "/home/zeshanmh/tanda/experiments/ddsm/data/json/mass_to_label.json"	

	X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_ddsm_data(data_dir=data_dir, \
		label_json=label_json, validation_set=True, segmentations=False)

	X_test = X_valid 
	Y_test = Y_valid

	dims = list(X_test.shape[1:])
	n_classes = Y_test.shape[-1]


	checkpoint_path = CHECKPOINT_PATHS[0]
	print "Loading saved model from %s..." % checkpoint_path
	D = GenericImageCNN(dims=dims)
	D.build_supervised(n_classes, 'D')

	with tf.device("/cpu:0"):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			D.restore(sess, checkpoint_path)

			print "Testing model on full test set..."
			acc, _ = D.get_accuracy(
				sess,
				X_test.reshape(-1, np.prod(dims)), 
				Y_test
			)
			print "Accuracy = %s" % acc

