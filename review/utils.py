import numpy as np 
import pandas as pd 
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision
import keras.backend as K
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, \
                            MultiHeadAttention, Dropout, LayerNormalization
import keras

from net import INCEPTION_Block

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
local_client_epochs = 1

def split_df(df):
    # read data from pickle
    df = df.sample(frac=1).reset_index(drop=True)

    # split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["type"])

    features = list(train_df.columns)
    features.remove("type")

    label_encoder = LabelEncoder()
    train_df["type"] = label_encoder.fit_transform(train_df["type"])
    test_df["type"] = label_encoder.transform(test_df["type"])

    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    X_train = train_df[features].values
    y_train = train_df["type"].values

    X_test = test_df[features].values
    y_test = test_df["type"].values

    clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    model = SelectFromModel(clf, prefit=True)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    X_train = X_train.reshape((-1, X_train.shape[-1], 1))
    X_test = X_test.reshape((-1, X_test.shape[-1], 1))

    return X_train, y_train, X_test, y_test, label_encoder

def load_data():
        # try to read data from pickle file
    try:
        df = pd.read_pickle("./data/df.pkl")
        print("Data loaded from pickle file")
    except:
        print("Data not found in pickle file, reading from csv file")
        benign_df = pd.read_csv('./data/5.benign.csv')
        g_c_df = pd.read_csv('./data/5.gafgyt.combo.csv')
        g_j_df = pd.read_csv('./data/5.gafgyt.junk.csv')
        g_s_df = pd.read_csv('./data/5.gafgyt.scan.csv')
        g_t_df = pd.read_csv('./data/5.gafgyt.tcp.csv')
        g_u_df = pd.read_csv('./data/5.gafgyt.udp.csv')
        m_a_df = pd.read_csv('./data/5.mirai.ack.csv')
        m_sc_df = pd.read_csv('./data/5.mirai.scan.csv')
        m_sy_df = pd.read_csv('./data/5.mirai.syn.csv')
        m_u_df = pd.read_csv('./data/5.mirai.udp.csv')
        m_u_p_df = pd.read_csv('./data/5.mirai.udpplain.csv')

        benign_df['type'] = 'benign'
        m_u_df['type'] = 'mirai_udp'
        g_c_df['type'] = 'gafgyt_combo'
        g_j_df['type'] = 'gafgyt_junk'
        g_s_df['type'] = 'gafgyt_scan'
        g_t_df['type'] = 'gafgyt_tcp'
        g_u_df['type'] = 'gafgyt_udp'
        m_a_df['type'] = 'mirai_ack'
        m_sc_df['type'] = 'mirai_scan'
        m_sy_df['type'] = 'mirai_syn'
        m_u_p_df['type'] = 'mirai_udpplain'

        df = pd.concat([benign_df, m_u_df, g_c_df,
                    g_j_df, g_s_df, g_t_df,
                    g_u_df, m_a_df, m_sc_df,
                    m_sy_df, m_u_p_df],
                    axis=0, sort=False, ignore_index=True)
        
        df.to_pickle("./data/df.pkl")
    
    return df

def get_class_weights(y_train):

    class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

    class_weights = {k: v for k,v in enumerate(class_weights)}

    return class_weights

def convert_to_categorical(y, nb_classes):
    return to_categorical(y, num_classes=nb_classes)

def iid_data_indices(labels: np.ndarray, nb_clients: int):
    data_len = len(labels)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    chunks = np.array_split(indices, nb_clients)
    return chunks

def non_iid_data_indices(labels: np.ndarray, nb_clients: int, nb_shards: int = 200):
    data_len = len(labels)
    indices = np.arange(data_len)
    indices = indices[labels.argsort()]
    shards = np.array_split(indices, nb_shards)
    random.shuffle(shards)
    shards_for_users = np.array_split(shards, nb_clients)
    indices_for_users = [np.hstack(x) for x in shards_for_users]
    return indices_for_users

def sample(y:np.ndarray, sampling_technique: str, nb_clients: int):

    if sampling_technique.lower() == "iid":
        sampler_fn = iid_data_indices
    else:
        sampler_fn = non_iid_data_indices
    client_data_indices = sampler_fn(y, nb_clients)
    return client_data_indices

def assign_data_to_clients(clients: dict, X:np.ndarray, y:np.ndarray, nb_classes:int, sampling_technique: str, X_train:np.ndarray, y_train:np.ndarray):
    sampled_data_indices = sample(y, sampling_technique, len(clients.keys()))
    for client_name, data_indices in zip(clients.keys(), sampled_data_indices):
        X = X_train[data_indices]
        y = y_train[data_indices]
        y = convert_to_categorical(y, nb_classes)
        clients[client_name] = list(zip(X, y))
    return clients

def create_clients(X, y, nb_classes, sampling_technique, num_clients=10, initial='clients'):
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    clients = {client_names[i] : [] for i in range(len(client_names))}
    return assign_data_to_clients(clients, X, y, nb_classes, sampling_technique, X, y)

def batch_data(data_shard, batch_size=64):
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    del data_shard

    len_label = len(label)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    del data, label
    return dataset.shuffle(len_label).batch(batch_size)

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.models.Sequential(
        [INCEPTION_Block(),
         INCEPTION_Block(),
         INCEPTION_Block()])(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def get_model(input_shape, nb_classes) -> tf.keras.Model:
    head_size=64                        # Embedding size for attention
    num_heads=3                         # Number of attention heads
    ff_dim=128                          # Hidden layer size in feed forward network inside transformer
    num_transformer_blocks=1
    mlp_units=[32]
    mlp_dropout=0.1
    dropout=0.1

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(nb_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

def calculate_global_count(clients):
    client_names = list(clients.keys())
    return sum([tf.data.experimental.cardinality(clients[client_name]["dataset"]).numpy() for client_name in client_names])

def weight_scalling_factor(clients, client_name, global_count):
    #get the bs
    bs = list(clients[client_name]["dataset"])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count_bs = global_count*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients[client_name]["dataset"]).numpy()*bs

    scaling_factor = local_count/global_count_bs

    print("weight {}: bs {} - global_count_bs {} - local_count {} - scaling_factor {}".format(client_name, bs, global_count_bs, local_count, scaling_factor))

    return scaling_factor

def get_model_weights(client_set, client_name, global_count):
    weight = client_set[client_name]["model"].get_weights()
    return weight

def average_model_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) / len(scaled_weight_list)
        avg_grad.append(layer_mean)
        
    return avg_grad

def train_client(client_name, global_weights, class_weights, client_set, comm_round):
    
    # client_set[client_name]["model"] = client_set[client_name]["model"]  - this is not needed (?)

    #set local model weight to the weight of the global model
    client_set[client_name]["model"].set_weights(global_weights)

    #fit local model with client's data
    print(f"[TRAINING] Round: {comm_round} | Client: {client_name}")
    client_set[client_name]["model"].fit(client_set[client_name]["dataset"], epochs=local_client_epochs, verbose=0, class_weight=class_weights)

    #scale the model weights and add to list
    # scaling_factor = weight_scalling_factor(client_set, client_name)
    # scaled_weights = scale_model_weights(client_set, client_name, global_count) 
    # scaled_local_weight_list.append(scaled_weights)
    # return scaled_weights

def test_model(X_test, y_test,  model, comm_round, mode, client_name = None, evaluation_scores = None):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    if (client_name == None):
        client_name = "all testing"

    #logits = model.predict(_X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(y_test, logits)
    y_hat = np.argmax(logits, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(logits, axis=1))
    
    r = Recall()
    r.update_state(y_test, logits)
    recall = r.result().numpy()
    
    p = Precision()
    p.update_state(y_test, logits)
    precision = p.result().numpy()
    
    f = f1_score(y_test, logits)
    f1 = f.numpy()

    if client_name != None and evaluation_scores != None:
        # append accuracy to list
        evaluation_scores[client_name] = loss
    
    print('mode: {} | comm_round: {} | global_loss: {} | global_accuracy: {:.4} | global_recall: {:.4} | global_precision: {:.4} | global_f1_score: {:.4} \n'.format(mode, comm_round, loss, accuracy, recall, precision, f1))
    return loss, accuracy, precision, recall, f1

def scale_model_weights_bk(client_set, client_name, global_count):
    '''function for scaling a models weights'''
    
    scaling_factor = weight_scalling_factor(client_set, client_name, global_count)
    weight = client_set[client_name]["model"].get_weights()

    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scaling_factor * weight[i])
    return weight_final