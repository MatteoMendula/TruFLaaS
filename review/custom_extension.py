import utils
import numpy as np
import matplotlib.pyplot as plt
import csv
import constants

def create_small_batches(clients_batched_standard, percentage_how_many_small, sample_indices):
    original_size = len(clients_batched_standard.keys())
    how_many_small = int(original_size * percentage_how_many_small)
    new_batches_with_small_clients = {}
    for index, client_name in enumerate(clients_batched_standard.keys()):
        # small clients
        if index in range(how_many_small):
            new_batches_with_small_clients[client_name] = [clients_batched_standard[client_name][index] for index in sample_indices]
        # normal clients
        else:        
            new_batches_with_small_clients[client_name] = clients_batched_standard[client_name]
    return new_batches_with_small_clients

def select_best_clients(client_set : dict, test_batched, comm_round, mode, std_factor = constants.std_factor):
    if mode != "TRUFLAAS" and mode != "TRUSTFED":
        print("[select_best_clients] mode error")
        return None

    client_names = list(client_set.keys())
    selected_clients = {}
    evaluation_scores = {}
    global_count = utils.calculate_global_count(client_set)

    local_weight_list = list()

    print("TESTING: ", mode)
    for client_name in client_names:
        model = client_set[client_name]["model"]
        for(x_batch, y_batch) in test_batched:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, model, comm_round, "local")
            evaluation_scores[client_name] = g_accuracy

    evaluation_scores_mean = np.mean(list(evaluation_scores.values()))
    evaluation_scores_std = np.std(list(evaluation_scores.values()))

    for client_name in evaluation_scores.keys():
        acc_score = evaluation_scores[client_name]
        if (mode == "TRUFLAAS" and acc_score > evaluation_scores_mean - std_factor * evaluation_scores_std):
            selected_clients[client_name] = client_set[client_name]
        elif (mode == "TRUSTFED" and acc_score > evaluation_scores_mean - std_factor * evaluation_scores_std and acc_score < evaluation_scores_mean + std_factor * evaluation_scores_std):
            selected_clients[client_name] = client_set[client_name]

    print("selected clients: ", selected_clients.keys())

    for client_name in selected_clients.keys():
        model_weights = utils.get_model_weights(selected_clients, client_name, global_count) 
        local_weight_list.append(model_weights)

    return local_weight_list

def select_all_clients(client_set, test_batched, comm_round):
    global_count = utils.calculate_global_count(client_set)
    client_names = list(client_set.keys())

    local_weight_list = list()

    # test anyway
    print("TESTING: ", "ALL")
    for client_name in client_names:
        model = client_set[client_name]["model"]
        for(x_batch, y_batch) in test_batched:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, model, comm_round, "local")

    for client_name in client_set.keys():
        model_weights = utils.get_model_weights(client_set, client_name, global_count) 
        local_weight_list.append(model_weights)

    return local_weight_list

def save_graphs(dict_of_metrics):
    colors = {
        "TRUFLAAS": "red",
        "TRUSTFED": "blue",
        "NO_SELECTION": "green"
    }
    for metric in constants.testing_metrics:
        for case in dict_of_metrics.keys():
            plt.plot(dict_of_metrics[case][metric], label="{}_{}".format(case, metric), color=colors[case])
        plt.legend()
        plt.savefig("./results/{}.png".format(metric))
        plt.show()

def save_csv(dict_of_metrics):
    header = ["round","loss_no_selection", "loss_truflaas", "loss_trustfed"]
    header += ["accuracy_no_selection", "accuracy_truflaas", "accuracy_trustfed"]
    header += ["precision_no_selection", "precision_truflaas", "precision_trustfed"]
    header += ["recall_no_selection", "recall_truflaas", "recall_trustfed"]
    header += ["f1_no_selection", "f1_truflaas", "f1_trustfed"]
    line = ', '.join(str(e) for e in header)
    with open("./results/out.csv", "w") as file:
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