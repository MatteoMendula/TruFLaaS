from keras.metrics import Recall, Precision
from utils import f1_score

import tensorflow as tf

learning_rate = 1e-2
comms_round = 5
num_clients = 30

std_factor = 0.8

metrics = ["loss", "accuracy", "precision", "recall", "f1"]

loss='categorical_crossentropy'
metrics = ["accuracy", Recall(), Precision(), f1_score]
optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)

BATCH_SIZE = 64
sampling_technique = "iid"

# ----------------- EXP 1 A -----------------
# how_small_percentage = 0.01
# percentage_small_clients = 1
# ----------------- EXP 1 B -----------------
how_small_percentage = 0.01
percentage_small_clients = 0.75