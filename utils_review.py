import pandas as pd
import os
import numpy as np
import torch
import csv
import copy
import random

import torch.multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

torch.multiprocessing.set_sharing_strategy('file_system')

def save_2d_matrix_to_csv_file(path, filename, row_list):
  create_folder_if_not_exists(path)
  with open(path + filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

def save_np_to_file(path, file_name, np_array):
    create_folder_if_not_exists(path)
    with open(path + file_name, 'wb') as f:
        np.save(f, np_array)

def process_data_final(device="cpu"):
    processed_data = process_data()

    print("Processing data...")
    print("process_data_final x_train shape", processed_data["x_train"].shape)

    x_train, y_train = processed_data["x_train"], processed_data["y_train"]
    x_test, y_test = processed_data["x_test"], processed_data["y_test"]
    x_train_no_rares, y_train_no_rares = processed_data["x_train_no_rares"], processed_data["y_train_no_rares"]
    x_train_yes_rares, y_train_yes_rares = processed_data["x_train_yes_rares"], processed_data["y_train_yes_rares"]
    x_test_no_rares, y_test_no_rares = processed_data["x_test_no_rares"], processed_data["y_test_no_rares"]
    x_test_yes_rares, y_test_yes_rares = processed_data["x_test_yes_rares"], processed_data["y_test_yes_rares"]

    # one hot labels encoding
    # y_train = torch.nn.functional.one_hot(torch.tensor(y_train)).float()
    # y_test = torch.nn.functional.one_hot(torch.tensor(y_test)).float()
    # y_train_no_rares = torch.nn.functional.one_hot(torch.tensor(y_train_no_rares)).float()
    # y_train_yes_rares = torch.nn.functional.one_hot(torch.tensor(y_train_yes_rares)).float()

    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()
    y_train_no_rares = torch.tensor(y_train_no_rares).long()
    y_train_yes_rares = torch.tensor(y_train_yes_rares).long()
    y_test_no_rares = torch.tensor(y_test_no_rares).long()
    y_test_yes_rares = torch.tensor(y_test_yes_rares).long()


    # transform to torch tensor - standard
    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    tensor_x_train = tensor_x_train.to(device)
    tensor_y_train = tensor_y_train.to(device)
    tensor_x_test = tensor_x_test.to(device)
    tensor_y_test = tensor_y_test.to(device)

    # training
    tensor_x_train_no_rares = torch.Tensor(x_train_no_rares)
    tensor_y_train_no_rares = torch.Tensor(y_train_no_rares)
    tensor_x_train_yes_rares = torch.Tensor(x_train_yes_rares)
    tensor_y_train_yes_rares = torch.Tensor(y_train_yes_rares)

    tensor_x_train_no_rares = tensor_x_train_no_rares.to(device)
    tensor_y_train_no_rares = tensor_y_train_no_rares.to(device)
    tensor_x_train_yes_rares = tensor_x_train_yes_rares.to(device)
    tensor_y_train_yes_rares = tensor_y_train_yes_rares.to(device)

    # testing 
    tensor_x_test_no_rares = torch.Tensor(x_test_no_rares)
    tensor_y_test_no_rares = torch.Tensor(y_test_no_rares)
    tensor_x_test_yes_rares = torch.Tensor(x_test_yes_rares)
    tensor_y_test_yes_rares = torch.Tensor(y_test_yes_rares)

    tensor_x_test_no_rares = tensor_x_test_no_rares.to(device)
    tensor_y_test_no_rares = tensor_y_test_no_rares.to(device)
    tensor_x_test_yes_rares = tensor_x_test_yes_rares.to(device)
    tensor_y_test_yes_rares = tensor_y_test_yes_rares.to(device)
    
    # create datasets for train and test 
    train_dataset = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    train_dataset_no_rares = torch.utils.data.TensorDataset(tensor_x_train_no_rares, tensor_y_train_no_rares)
    train_dataset_yes_rares = torch.utils.data.TensorDataset(tensor_x_train_yes_rares, tensor_y_train_yes_rares)

    test_dataset = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_dataset_no_rares= torch.utils.data.TensorDataset(tensor_x_test_no_rares, tensor_y_test_no_rares)
    test_dataset_yes_rares = torch.utils.data.TensorDataset(tensor_x_test_yes_rares, tensor_y_test_yes_rares)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=120, shuffle=True)
    train_loader_no_rares = torch.utils.data.DataLoader(train_dataset_no_rares, batch_size=200, shuffle=True)
    train_loader_yes_rares = torch.utils.data.DataLoader(train_dataset_yes_rares, batch_size=200, shuffle=True)
    test_loader_no_rares = torch.utils.data.DataLoader(test_dataset_no_rares, batch_size=120, shuffle=True)
    test_loader_yes_rares = torch.utils.data.DataLoader(test_dataset_yes_rares, batch_size=120, shuffle=True)

    # experiment 1: ci sono nodi che hanno piu' dati degli altri, trustFed li escluderebbe, noi no
    train_loader_small = torch.utils.data.DataLoader(train_dataset, batch_size=30,shuffle=True)
    test_loader_small = torch.utils.data.DataLoader(test_dataset, batch_size=15,shuffle=True)
    train_loader_big = torch.utils.data.DataLoader(train_dataset, batch_size=350, shuffle=True)
    test_loader_big = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True)

    final_data = {}
    # print("shape train_loader", train_loader.dataset.tensors[0].shape[-1])
    final_data["train_loader"] = train_loader
    final_data["test_loader"] = test_loader
    final_data["train_loader_small"] = train_loader_small
    final_data["test_loader_small"] = test_loader_small
    final_data["train_loader_big"] = train_loader_big
    final_data["test_loader_big"] = test_loader_big
    final_data["train_loader_no_rares"] = train_loader_no_rares
    final_data["train_loader_yes_rares"] = train_loader_yes_rares
    final_data["test_loader_no_rares"] = test_loader_no_rares
    final_data["test_loader_yes_rares"] = test_loader_yes_rares

    return final_data

def select_node_to_discard_trustfed(result):
  outliers = []
  data_std = np.std(result)
  data_mean = np.mean(result)
  anomaly_cut_off = data_std * 2
  lower_limit = data_mean - anomaly_cut_off
  upper_limit = data_mean + anomaly_cut_off

  for index, loss in enumerate(result):
    if loss > upper_limit or loss < lower_limit:
        outliers.append(index)
        
  return outliers

def select_node_to_discard_truflass(result):
  outliers = []
  data_std = np.std(result)
  data_mean = np.mean(result)
  anomaly_cut_off = data_std * 2
  upper_limit = data_mean + anomaly_cut_off

  # only major losses are detected
  # low losses are accepted because no forging is possible

  for index, loss in enumerate(result):
    if loss > upper_limit:
        outliers.append(index)
        
  return outliers


# FED AVERAGE WEIGHTED
def aggregate_model_weighted(models, memory, iteration, iterations, device = "cpu"):
    if device != "cpu":
      return aggregate_model_cuda_weighted(models, device)
    # no cuda
    model_aggregated = []
    for param in models[0][0].parameters():
      model_aggregated += [np.zeros(param.shape)]

    sum_weights = 0
    for model in models:
      select = random.randint(0, 1)
      i = 0
      model_model = model[0]
      model_id = model[1]
      weight = 1
      # if select == 0:
      if model_id in memory.keys():
        weight = 1 - ( memory[model_id] / (iteration + 1) )
        print(f"{model_id}) weight={weight}")
      sum_weights += weight
      for param in model_model.parameters():
        model_aggregated[i] += param.detach().numpy() * weight
        i += 1

    print("sum_weights", sum_weights)

    model_aggregated = np.array(model_aggregated, dtype=object) / sum_weights
    
    return model_aggregated

# FED AVERAGE LIKE AGGREGATION
def aggregate_model(models, device="cpu"):
    # if device != "cpu":
    #   return aggregate_model_cuda(models, device)

    # no cuda
    model_aggregated = []
    for param in models[0].parameters():
      model_aggregated += [np.zeros(param.shape)]

    for model in models:
      i = 0
      for param in model.parameters():
        model_aggregated[i] += param.detach().cpu().numpy() * 1
        i += 1

    model_aggregated = np.array(model_aggregated, dtype=object) / len(models)
    
    return model_aggregated

def aggregate_model_cuda_weighted(models, device):
    model_aggregated = torch.FloatTensor(
      list(models[0][0].parameters())
    )
    print("model_aggregated.shape", model_aggregated.shape)
    _models = models[1:]
    for model in _models:
      this_model_params = list(model[0].parameters())
      model_aggregated = torch.add(model_aggregated, this_model_params)

    model_aggregated = torch.div(model_aggregated,len(models))

    return model_aggregated

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def process_data():
  dirname = os.getcwd()
  folder_path = os.path.join(dirname, '')
  df_path = os.path.join(folder_path, './data/review.pkl')
  df = pd.read_pickle(df_path)
  
  # split normal data
  print("splitting normal data ------------")
  X_train, y_train, X_test, y_test, label_encoder, rares_index = split_df(df)
  
  print("X_train type", type(X_train))
  print("X_train.shape", X_train.shape)
  print("y_train.shape", y_train.shape)
  print("X_test.shape", X_test.shape)
  print("y_test.shape", y_test.shape)

  # split rare data
  print("splitting rare data ------------")
  X_train_rare, y_train_rare, X_test_rare, y_test_rare = get_rare_cases_from_df(X_train, y_train, X_test, y_test, rares_index)
  
  print("X_train type", type(X_train_rare))
  print("X_train_rare.shape", X_train_rare.shape)
  print("y_train_rare.shape", y_train_rare.shape)
  print("X_test_rare.shape", X_test_rare.shape)
  print("y_test_rare.shape", y_test_rare.shape)

  # split no_rare data
  print("splitting no rare data ------------")
  X_train_no_rare, y_train_no_rare, X_test_no_rare, y_test_no_rare = remove_rare_cases_from_df(X_train, y_train, X_test, y_test, rares_index)

  print("X_train_no_rare.shape", X_train_no_rare.shape)
  print("y_train_no_rare.shape", y_train_no_rare.shape)
  print("X_test_no_rare.shape", X_test_no_rare.shape)
  print("y_test_no_rare.shape", y_test_no_rare.shape)

  processed_data = {}
  processed_data["n_classes"] = len(label_encoder.classes_)
  processed_data["x_train"] = X_train
  processed_data["y_train"] = y_train
  processed_data["x_test"] = X_test
  processed_data["y_test"] = y_test

  processed_data["x_train_yes_rares"] = X_train_rare
  processed_data["y_train_yes_rares"] = y_train_rare

  processed_data["x_train_no_rares"] = X_train_no_rare
  processed_data["y_train_no_rares"] = y_train_no_rare

  processed_data["x_test_no_rares"] = X_test_no_rare
  processed_data["x_test_yes_rares"] = X_test_rare
  
  processed_data["y_test_yes_rares"] = y_test_rare
  processed_data["y_test_no_rares"] = y_test_no_rare

  return processed_data


def get_rare_cases_from_df(X_train, y_train, X_test, y_test, rares_index):
  _X_train = []
  _y_train = []
  _X_test = []
  _y_test = []

  for i in range(len(X_train)):
    if y_train[i] == rares_index:
      _X_train.append(X_train[i])
      _y_train.append(y_train[i])

  for i in range(len(X_test)):
    if y_test[i] == rares_index:
      _X_test.append(X_test[i])
      _y_test.append(y_test[i])

  _X_train = np.array(_X_train)
  _y_train = np.array(_y_train)
  _X_test = np.array(_X_test)
  _y_test = np.array(_y_test)

  return _X_train, _y_train, _X_test, _y_test

def remove_rare_cases_from_df(X_train, y_train, X_test, y_test, rares_index):
  _X_train = []
  _y_train = []
  _X_test = []
  _y_test = []

  for i in range(len(X_train)):
    if y_train[i] != rares_index:
      _X_train.append(X_train[i])
      _y_train.append(y_train[i])

  for i in range(len(X_test)):
    if y_test[i] != rares_index:
      _X_test.append(X_test[i])
      _y_test.append(y_test[i])

  _X_train = np.array(_X_train)
  _y_train = np.array(_y_train)
  _X_test = np.array(_X_test)
  _y_test = np.array(_y_test)

  return _X_train, _y_train, _X_test, _y_test

def split_df(df):
  # read data from pickle
  df = df.sample(frac=1).reset_index(drop=True)

  # split data into train and test
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["type"])

  # get the list of columns except the type column
  features = list(train_df.columns)
  features.remove("type")

  # encode labels in the same way for train and test
  label_encoder = LabelEncoder()
  train_df["type"] = label_encoder.fit_transform(train_df["type"])
  test_df["type"] = label_encoder.transform(test_df["type"])
  rares_index = train_df["type"].value_counts().nsmallest(1).index.values.astype(int)[0]


  # scale data in the same way for train and test
  scaler = MinMaxScaler()
  train_df[features] = scaler.fit_transform(train_df[features])
  test_df[features] = scaler.transform(test_df[features])

  # get all the values from all the columns except the type column
  X_train = train_df[features].values
  y_train = train_df["type"].values

  # get all the values from the type column
  X_test = test_df[features].values
  y_test = test_df["type"].values

  clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
  clf = clf.fit(X_train, y_train)
  model = SelectFromModel(clf, prefit=True)
  X_train = model.transform(X_train)
  X_train = X_train.reshape((-1, 1, X_train.shape[-1]))

  X_test = model.transform(X_test)
  X_test = X_test.reshape((-1, 1, X_test.shape[-1]))

  # y_train = y_train.reshape((y_train.shape[-1], -1))
  # y_test = y_test.reshape((y_test.shape[-1], -1))

  # print("X_TRAiN:100", X_train[:100])

  return X_train, y_train, X_test, y_test, label_encoder, rares_index