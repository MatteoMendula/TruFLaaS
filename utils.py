import pandas as pd
import os
import numpy as np
import torch
import csv
import copy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

WINDOW_SIZE = 1
PERCENTILES_ON_TRAINING = (18.262574724926324, 195.97552938214926)
PERCENTILES_ON_TESTING = (64.5643214552666, 216.7330157866137)

def save_2d_matrix_to_csv_file(filename, row_list):
  with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)

def save_np_to_file(file_name, np_array):
    with open(f'{file_name}.npy', 'wb') as f:
        np.save(f, np_array)

def process_data():
  # load the data
  dirname = os.getcwd()
  folder_path = os.path.join(dirname, '')
  
  train_data_path = os.path.join(folder_path, './data/train_data_FD001.txt')
  test_data_path = os.path.join(folder_path, './data/test_data_FD001.txt')
  train_data_path_pkl = os.path.join(folder_path, './data/train_data_FD001.pkl')
  test_data_path_pkl = os.path.join(folder_path, './data/test_data_FD001.pkl')
  train_data, test_data = None, None

  if os.path.exists(train_data_path_pkl):
    train_data = pd.read_pickle( train_data_path_pkl )
  else:
    train_data = pd.read_csv(train_data_path)
    train_data.set_index('time_in_cycles')
    train_data.to_pickle("./data/train_data_FD001.pkl")
  
  if os.path.exists(test_data_path_pkl):
    test_data = pd.read_pickle( test_data_path_pkl )
  else:
    test_data = pd.read_csv(test_data_path)
    test_data.set_index('time_in_cycles')
    test_data.to_pickle("./data/test_data_FD001.pkl")

  # retrieve the max cycles per engine: RUL
  train_rul = pd.DataFrame(train_data.groupby('engine_no')['time_in_cycles'].max()).reset_index()

  # merge the RULs into the training data
  train_rul.columns = ['engine_no', 'max']
  train_data = train_data.merge(train_rul, on=['engine_no'], how='left')
  # add the current RUL for every cycle
  train_data['RUL'] = train_data['max'] - train_data['time_in_cycles']
  train_data.drop('max', axis=1, inplace=True)

  # analyze RUL distribution
  # sns.displot(train_data['RUL'])

  # drop the columns not needed
  cols_nan = train_data.columns[train_data.isna().any()].tolist()
  cols_const = [ col for col in train_data.columns if len(train_data[col].unique()) <= 2 ]

  cols_irrelevant = ['operational_setting_1', 'operational_setting_2', 'sensor_measurement_11', 'sensor_measurement_12', 'sensor_measurement_13']

  # Drop the columns without or with constant data
  train_data = train_data.drop(columns=cols_const + cols_nan + cols_irrelevant)
  test_data = test_data.drop(columns=cols_const + cols_nan + cols_irrelevant)

  train_data_yes_rares = train_data[train_data['RUL'] < PERCENTILES_ON_TRAINING[0]]
  train_data_no_rares = train_data[train_data['RUL'] > PERCENTILES_ON_TRAINING[0]]
  test_data_yes_rares = test_data[test_data['RUL'] < PERCENTILES_ON_TESTING[0]]
  test_data_no_rares = test_data[test_data['RUL'] > PERCENTILES_ON_TESTING[0]]

  processed_data = {}
  processed_data["train_data"] = train_data
  processed_data["test_data"] = test_data
  processed_data["train_data_yes_rares"] = train_data_yes_rares
  processed_data["train_data_no_rares"] = train_data_no_rares
  processed_data["test_data_yes_rares"] = test_data_yes_rares
  processed_data["test_data_no_rares"] = test_data_no_rares

  return processed_data

def transform_to_windowed_data(dataset, window_size, window_limit = 0, verbose = False):

  features = []
  labels = []

  dataset = dataset.set_index('time_in_cycles')
  data_per_engine = dataset.groupby('engine_no')

  for engine_no, engine_data in data_per_engine:
      # skip if the engines cycles are too few
      if len(engine_data) < window_size + window_limit -1:
        continue

      if window_limit != 0:
        window_count = window_limit
      else:
        window_count = len(engine_data) - window_size

      for i in range(0, window_count):
        # take the last x cycles where x is the window size
        start = -window_size - i
        end = len(engine_data) - i
        inputs = engine_data.iloc[start:end]

        # use the RUL of the last cycle as label
        outputs = engine_data.iloc[end - 1, -1]

        inputs = inputs.drop(['engine_no', 'RUL'], axis=1)

        features.append(inputs.values)
        labels.append(outputs)

  features = np.array(features)
  labels = np.array(labels)
  labels = np.expand_dims(labels, axis=1)

  if verbose:
    print("{} features with shape {}".format(len(features), features[0].shape))
    print("{} labels with shape {}".format(len(labels), labels.shape))

  return features, labels


def process_data_final(device="cpu"):
    processed_data = process_data()
    x_train, y_train = transform_to_windowed_data(processed_data["train_data"], WINDOW_SIZE)
    x_test, y_test = transform_to_windowed_data(processed_data["test_data"], WINDOW_SIZE)
    x_train_no_rares, y_train_no_rares = transform_to_windowed_data(processed_data["train_data_no_rares"], WINDOW_SIZE)
    x_train_yes_rares, y_train_yes_rares = transform_to_windowed_data(processed_data["train_data_yes_rares"], WINDOW_SIZE)
    x_test_no_rares, y_test_no_rares = transform_to_windowed_data(processed_data["test_data_no_rares"], WINDOW_SIZE)
    x_test_yes_rares, y_test_yes_rares = transform_to_windowed_data(processed_data["test_data_yes_rares"], WINDOW_SIZE)

    # clip RUL values    # ----------------------------- sta togliemdo i valori superiori a 110!
    rul_clip_limit = 110
    y_train_cliped = y_train.clip(max=rul_clip_limit)
    y_test_cliped = y_test.clip(max=rul_clip_limit)

    # y_test = y_test.clip(max=rul_clip_limit)
    # y_test_no_rares = y_test_no_rares.clip(max=rul_clip_limit)
    # y_test_yes_rares = y_test_yes_rares.clip(max=rul_clip_limit)

    # transform to torch tensor - standard
    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_y_train_cliped = torch.Tensor(y_train_cliped)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)
    tensor_y_test_cliped = torch.Tensor(y_test_cliped)

    tensor_x_train = tensor_x_train.to(device)
    tensor_y_train = tensor_y_train.to(device)
    tensor_y_train_cliped = tensor_y_train_cliped.to(device)
    tensor_x_test = tensor_x_test.to(device)
    tensor_y_test = tensor_y_test.to(device)
    tensor_y_test_cliped = tensor_y_test_cliped.to(device)

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

    #  Data Normalization
    train_mean = tensor_x_train.mean(0)
    train_std = tensor_x_train.std(0)
    tensor_x_train = (tensor_x_train - train_mean) / train_std
    tensor_x_train = tensor_x_train.to(device)

    test_mean = tensor_x_test.mean(0)
    test_std = tensor_x_test.std(0)
    tensor_x_test = (tensor_x_test - test_mean) / test_std
    tensor_x_test = tensor_x_test.to(device)

    # training
    x_train_no_rares_mean = tensor_x_train_no_rares.mean(0)
    x_train_no_rares_std = tensor_x_train_no_rares.std(0)
    tensor_x_train_no_rares = (tensor_x_train_no_rares - x_train_no_rares_mean) / x_train_no_rares_std
    tensor_x_train_no_rares = tensor_x_train_no_rares.to(device)

    x_train_yes_rares_mean = tensor_x_train_yes_rares.mean(0)
    x_train_yes_rares_std = tensor_x_train_yes_rares.std(0)
    tensor_x_train_yes_rares = (tensor_x_train_yes_rares - x_train_yes_rares_mean) / x_train_yes_rares_std
    tensor_x_train_yes_rares = tensor_x_train_yes_rares.to(device)

    # testing
    x_test_no_rares_mean = tensor_x_test_no_rares.mean(0)
    x_test_no_rares_std = tensor_x_test_no_rares.std(0)
    tensor_x_test_no_rares = (tensor_x_test_no_rares - x_test_no_rares_mean) / x_test_no_rares_std
    tensor_x_test_no_rares = tensor_x_test_no_rares.to(device)

    x_test_yes_rares_mean = tensor_x_test_yes_rares.mean(0)
    x_test_yes_rares_std = tensor_x_test_yes_rares.std(0)
    tensor_x_test_yes_rares = (tensor_x_test_yes_rares - x_test_yes_rares_mean) / x_test_yes_rares_std
    tensor_x_test_yes_rares = tensor_x_test_yes_rares.to(device)

    # create datasets for train and test 
    train_dataset = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    train_dataset_cliped = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train_cliped)
    train_dataset_no_rares = torch.utils.data.TensorDataset(tensor_x_train_no_rares, tensor_y_train_no_rares)
    train_dataset_yes_rares = torch.utils.data.TensorDataset(tensor_x_train_yes_rares, tensor_y_train_yes_rares)

    # print("tensor_x_train_yes_rares.size", tensor_x_train_yes_rares.size())
    # print("tensor_y_train_yes_rares.size", tensor_y_train_yes_rares.size())
    # print("------------------")
    # print("tensor_x_test_yes_rares.size", tensor_x_test_yes_rares.size())
    # print("tensor_y_test_yes_rares.size", tensor_y_test_yes_rares.size())

    test_dataset = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    test_dataset_cliped = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test_cliped)
    test_dataset_no_rares= torch.utils.data.TensorDataset(tensor_x_test_no_rares, tensor_y_test_no_rares)
    test_dataset_yes_rares = torch.utils.data.TensorDataset(tensor_x_test_yes_rares, tensor_y_test_yes_rares)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=120, shuffle=True)
    train_loader_cliped = torch.utils.data.DataLoader(train_dataset_cliped, batch_size=200, shuffle=True)
    test_loader_cliped = torch.utils.data.DataLoader(test_dataset_cliped, batch_size=120, shuffle=True)
    train_loader_no_rares = torch.utils.data.DataLoader(train_dataset_no_rares, batch_size=200, shuffle=True)
    train_loader_yes_rares = torch.utils.data.DataLoader(train_dataset_yes_rares, batch_size=200, shuffle=True)
    test_loader_no_rares = torch.utils.data.DataLoader(test_dataset_no_rares, batch_size=120, shuffle=True)
    test_loader_yes_rares = torch.utils.data.DataLoader(test_dataset_yes_rares, batch_size=120, shuffle=True)

    # experiment 1: ci sono nodi che hanno piu' dati degli altri, trustFed li escluderebbe, noi no
    train_loader_small = torch.utils.data.DataLoader(train_dataset, batch_size=30,shuffle=True)
    test_loader_small = torch.utils.data.DataLoader(test_dataset, batch_size=15,shuffle=True)
    train_loader_big = torch.utils.data.DataLoader(train_dataset, batch_size=350, shuffle=True)
    test_loader_big = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True)

    train_loader_small_cliped = torch.utils.data.DataLoader(train_dataset_cliped, batch_size=30,shuffle=True)
    test_loader_small_cliped = torch.utils.data.DataLoader(test_dataset_cliped, batch_size=15,shuffle=True)
    train_loader_big_cliped = torch.utils.data.DataLoader(train_dataset_cliped, batch_size=350, shuffle=True)
    test_loader_big_cliped = torch.utils.data.DataLoader(test_dataset_cliped, batch_size=200, shuffle=True)

    final_data = {}
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

    final_data["train_loader_small_cliped"] = train_loader_small_cliped
    final_data["test_loader_small_cliped"] = test_loader_small_cliped
    final_data["train_loader_big_cliped"] = train_loader_big_cliped
    final_data["test_loader_big_cliped"] = test_loader_big_cliped
    final_data["train_loader_cliped"] = train_loader_cliped
    final_data["test_loader_cliped"] = test_loader_cliped

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
def aggregate_model_weighted(models, memory, iteration, device):
    if device != "cpu":
      return aggregate_model_cuda_weighted(models, device)
    # no cuda
    model_aggregated = []
    for param in models[0][0].parameters():
      model_aggregated += [np.zeros(param.shape)]

    sum_weights = 0
    for model in models:
      i = 0
      model_model = model[0]
      model_id = model[1]
      weight = 1
      if model_id in memory.keys():
        weight = 1 - ( memory[model_id] / (iteration + 1) )
      sum_weights += weight
      print(f"{model_id}) weight={weight}")
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