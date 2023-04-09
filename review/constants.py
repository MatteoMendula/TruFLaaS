from keras.metrics import Recall, Precision
from utils import f1_score

import tensorflow as tf

learning_rate = 1e-2
comms_round = 10
num_clients = 30
local_testing_size = 0.01

std_factor = 0.8

testing_metrics = ["loss", "accuracy", "precision", "recall", "f1"]

loss='categorical_crossentropy'
metrics = ["accuracy", Recall(), Precision(), f1_score]
optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

BATCH_SIZE = 64
sampling_technique = "iid"
