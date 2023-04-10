import utils
import numpy as np
import matplotlib.pyplot as plt
import csv
import constants
from threading import Thread
import tensorflow as tf
import random
import os

def create_small_batches(clients_batched_standard, special_clients, original_sample_size, small_sample_size):
    new_batches_with_small_clients = {}

    for _, client_name in enumerate(clients_batched_standard.keys()):
        sample_indices = np.random.choice(original_sample_size, small_sample_size, replace=False)
        # small clients
        if client_name in special_clients:
            new_batches_with_small_clients[client_name] = [clients_batched_standard[client_name][index] for index in sample_indices]
        # normal clients
        else:        
            new_batches_with_small_clients[client_name] = clients_batched_standard[client_name]
    return new_batches_with_small_clients

def create_noisy_batches(clients_batches_original, clients_batches_malicous, special_clients, sample_indices):
    # to run faster every client is how_small_percentage times smaller
    new_batches_with_malicious_clients = {}

    # make how_many_noisy clients noisy
    for index, client_name in enumerate(clients_batches_original.keys()):
        # malicious clients
        if client_name in special_clients:
            new_batches_with_malicious_clients[client_name] = clients_batches_malicous[client_name]
        else:
            new_batches_with_malicious_clients[client_name] = clients_batches_original[client_name]

    # reduce the size of all clients
    for client_name in new_batches_with_malicious_clients.keys():
        new_batches_with_malicious_clients[client_name] = [new_batches_with_malicious_clients[client_name][index] for index in sample_indices]

    return new_batches_with_malicious_clients

def create_batches_with_no_rares(clients_batches_original, clients_batches_no_rares, percentage_how_many_no_rares, sample_indices):
    # to run faster every client is how_small_percentage times smaller
    number_of_clients = len(clients_batches_original.keys())
    how_many_no_rares = int(number_of_clients * percentage_how_many_no_rares)
    new_batches_with_no_rares_clients = {}

     # make how_many_noisy clients noisy
    for index, client_name in enumerate(clients_batches_original.keys()):
        # malicious clients
        if index in range(how_many_no_rares):
            new_batches_with_no_rares_clients[client_name] = clients_batches_no_rares[client_name]
        else:
            new_batches_with_no_rares_clients[client_name] = clients_batches_original[client_name]

    # reduce the size of all clients
    for client_name in new_batches_with_no_rares_clients.keys():
        new_batches_with_no_rares_clients[client_name] = [new_batches_with_no_rares_clients[client_name][index] for index in sample_indices]

    return new_batches_with_no_rares_clients

def select_best_clients(client_set : dict, test_batched, comm_round, mode, experiment_name, test_batch_rares = None):
    if mode != "TRUFLAAS" and mode != "TRUSTFED" and mode != "UNION" and mode != "INTERSECTION":    
        print("[select_best_clients] mode error")
        return None

    client_names = list(client_set.keys())
    selected_clients = {}
    global_count = utils.calculate_global_count(client_set)

    local_weight_list = list()

    discarding_votes = {}
    evaluation_scores = {}
    threads = [None] * len(client_names)

    print(" ------------------------ ")
    print("TESTING: ", mode)
    print(" ------------------------ ")
    for i, client_name_tester_name in enumerate(client_names):
        if mode == "TRUSTFED":
            threads[i] = Thread(target=test_other_clients_trustfed, args=(client_set, client_names, client_name_tester_name, mode, comm_round, discarding_votes, experiment_name))
            threads[i].start()
        elif mode == "TRUFLAAS":
            client_to_test_name = client_name_tester_name
            threads[i] = Thread(target=test_truflass_clients, args=(client_set, client_to_test_name, test_batched, mode, comm_round, evaluation_scores, experiment_name))
            threads[i].start()
            
    for i in range(len(threads)):
        threads[i].join()
    
    if mode == "TRUSTFED":
        for client_name in discarding_votes.keys():
            if discarding_votes[client_name] > int(len(client_names)/2):
                print("discarding_votes[{}]".format(client_name), discarding_votes[client_name])
                print("skipped: ", client_name)
                continue
            selected_clients[client_name] = client_set[client_name]

    elif mode == "TRUFLAAS":
        evaluation_scores_mean = np.mean(list(evaluation_scores.values()))
        evaluation_scores_std = np.std(list(evaluation_scores.values()))
        print("evaluation_scores_mean: ", evaluation_scores_mean)
        print("evaluation_scores_std: ", evaluation_scores_std)
        for client_name in evaluation_scores.keys():
            print("evaluation_scores[{}]: ".format(client_name), evaluation_scores[client_name])
            if evaluation_scores[client_name] > evaluation_scores_mean + constants.std_factor * evaluation_scores_std:
                print("skipped: ", client_name)
                continue
            selected_clients[client_name] = client_set[client_name]

    print("selected clients: ", selected_clients.keys())

    for client_name in selected_clients.keys():
        model_weights = utils.get_model_weights(selected_clients, client_name, global_count) 
        local_weight_list.append(model_weights)

    return local_weight_list

def select_all_clients(client_set, test_batched, comm_round, experiment_name):
    global_count = utils.calculate_global_count(client_set)

    local_weight_list = list()
    print("TESTING: ", "SELECT ALL")
    for client_name in client_set.keys():
        model_weights = utils.get_model_weights(client_set, client_name, global_count) 
        local_weight_list.append(model_weights)

    return local_weight_list

def save_graphs(dict_of_metrics, experiment_name, percentage_small_clients):
    print("dict_of_metrics", dict_of_metrics)
    print("experiment_name", experiment_name)
    print("percentage_small_clients", percentage_small_clients)

    case_name = "case_{}".format(str(percentage_small_clients))
    colors = {
        "TRUFLAAS": "red",
        "TRUSTFED": "blue",
        "NO_SELECTION": "green",

        "UNION": "red",
        "INTERSECTION": "blue"
    }
    for metric in constants.testing_metrics:
        plt.clf()
        for case in dict_of_metrics.keys():
            # case = "TRUFLAAS"/"TRUSTFED"/"NO_SELECTION"
            plt.plot(dict_of_metrics[case][metric], label="{}_{}".format(case, metric), color=colors[case])
        plt.legend()
        folder_path = "./results/{}/{}".format(experiment_name, case_name)
        create_folder_if_not_exists(folder_path)
        plt.savefig("./results/{}/{}/{}.png".format(experiment_name, case_name, metric))
        # plt.show()

def save_csv(dict_of_metrics, experiment_name, percentage_small_clients, is_union_intersection = False):
    print("dict_of_metrics", dict_of_metrics)
    print("experiment_name", experiment_name)
    print("percentage_small_clients", percentage_small_clients)

    case_name = "case_{}".format(str(percentage_small_clients))
    if is_union_intersection == False:
        header = ["round","loss_no_selection", "loss_truflaas", "loss_trustfed"]
        header += ["accuracy_no_selection", "accuracy_truflaas", "accuracy_trustfed"]
        header += ["precision_no_selection", "precision_truflaas", "precision_trustfed"]
        header += ["recall_no_selection", "recall_truflaas", "recall_trustfed"]
        header += ["f1_no_selection", "f1_truflaas", "f1_trustfed"]
    else:
        header = ["round","loss_no_selection", "loss_union", "loss_intersection"]
        header += ["accuracy_no_selection", "accuracy_union", "accuracy_intersection"]
        header += ["precision_no_selection", "precision_union", "precision_intersection"]
        header += ["recall_no_selection", "recall_union", "recall_intersection"]
        header += ["f1_no_selection", "f1_union", "f1_intersection"]
    line = ', '.join(str(e) for e in header)
    folder_path = "./results/{}/{}".format(experiment_name, case_name)
    create_folder_if_not_exists(folder_path)
    with open("{}/out.csv".format(folder_path), "w") as file:
        file.write(line+"\n")
        for round in range(constants.comms_round):
            line = "{}, ".format(round)
            for metric in constants.testing_metrics:
                for case in dict_of_metrics.keys():
                    line += "{}, ".format(dict_of_metrics[case][metric][round])
            file.write(line+"\n")

def sample_test(X_test, y_test, sample_percentage=0.2):
    sample_size = int(len(X_test)*sample_percentage)
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_test_sample = X_test[sample_indices]
    y_test_sample = y_test[sample_indices]
    return X_test_sample, y_test_sample

def create_testing_batched(X_test, y_test):
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(X_test))
    return test_batched

def get_rare_cases_from_df(df):
    return df[df["type"] == "gafgyt_scan"]

def remove_rare_cases_from_df(df):
    return df[df["type"] != "gafgyt_scan"]

def create_noisy_df(df):
    df_only_data = df.copy()
    df_only_data.drop(['type'], axis = 1, inplace = True) 
    mu, sigma = 0, 0.5
    noise = np.random.normal(mu, sigma, df_only_data.shape) 
    noisy_df = df_only_data + noise
    noisy_df['type'] = df['type']
    return noisy_df

def test_other_clients_trustfed(client_set, client_names, client_name_tester_name, mode, comm_round, discarding_votes, experiment_name):
    evaluation_scores = {}
    for client_name in client_names:
        # a client cannot test itself
        if client_name == client_name_tester_name:
            continue
        model = client_set[client_name]["model"]
        test_batched_client = client_set[client_name_tester_name]["testing"]
        # for (x_batch, y_batch) in test_batched:
        for (x_batch, y_batch) in test_batched_client:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, model, comm_round, "[{}]local_{}".format(experiment_name, mode))
            evaluation_scores[client_name] = np.round(g_loss, 3)

    evaluation_scores_mean = np.mean(list(evaluation_scores.values()))
    evaluation_scores_std = np.std(list(evaluation_scores.values()))

    for client_name in client_names:
        if client_name == client_name_tester_name:
            continue
        if client_name not in discarding_votes.keys():
            discarding_votes[client_name] = 0
        # if mode == "TRUFLAAS" and evaluation_scores[client_name] < evaluation_scores_mean - constants.std_factor * evaluation_scores_std:
        #     discarding_votes[client_name] += 1
        elif mode == "TRUSTFED" and (evaluation_scores[client_name] < evaluation_scores_mean - constants.std_factor * evaluation_scores_std or evaluation_scores[client_name] > evaluation_scores_mean + constants.std_factor * evaluation_scores_std):
            discarding_votes[client_name] += 1

def test_truflass_clients(client_set, client_to_test_name, test_batched, mode, comm_round, evaluation_scores, experiment_name):
    model = client_set[client_to_test_name]["model"]
    for (x_batch, y_batch) in test_batched:
        g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, model, comm_round, "[{}]local_{}".format(experiment_name, mode))
        evaluation_scores[client_to_test_name] = np.round(g_loss, 3)

def split_x_y_into_chunks(x, y, n_chunks):
    if len(x) != len(y):
        return "error"
    def randomize_indexes(l, parts):
        list_len = len(l)
        list_indexes = list(range(list_len))
        random.shuffle(list_indexes)
        return list_indexes

    def chunkify_indexes(lst,n):
        return [lst[i::n] for i in range(n)]
    
    indexes = randomize_indexes(x, 10)
    indexes_chunks = chunkify_indexes(indexes, n_chunks)
    my_structure = {}
    print("len(indexes_chunks)", len(indexes_chunks))
    print("len(indexes_chunks)[0]", len(indexes_chunks[0]))
    for index in range(len(indexes_chunks)):
        my_structure[index] = {}
        my_structure[index]["x"] = []
        my_structure[index]["y"] = []
        for el in indexes_chunks[index]:
            my_structure[index]["x"].append(x[el])
            my_structure[index]["y"].append(y[el])
    return my_structure

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)