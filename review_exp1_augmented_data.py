import torch
import time

import matplotlib.pyplot as plt
import numpy as np
import copy
np.random.seed(1)

from worker import Worker
from net_review import Net
from utils_review import aggregate_model, process_data_final, select_node_to_discard_trustfed, select_node_to_discard_truflass, save_np_to_file, save_2d_matrix_to_csv_file, create_folder_if_not_exists
import random

from playsound import playsound as play

experiments = 10
iteration = 20
size = 20
num_workers = 100
n_validators = 100
rounds = 100
special_options = [40, 25, 0]

def experiment_rare_cases(experiment_counter, n_specials, final_data):

    n_augmented = n_specials
    all_trainers = range(num_workers)
    augmented_nodes = random.sample(all_trainers, n_specials)
    no_augmented_nodes = list(set(all_trainers) - set(augmented_nodes))
    print("-----------------------------------")
    print("augmented_nodes", augmented_nodes)
    print("-----------------------------------")
    print("no_augmented_nodes", no_augmented_nodes)

    model_original = Net(input_shape = final_data["train_loader"].dataset.tensors[0].shape[-1])
    validator_workers = {}
    workers_no_augmented = {}
    workers_yes_augmented_no_filter = {}
    workers_yes_augmented_trustfed = {}
    workers_yes_augmented_truflass = {}

    train_standard = [(data, target) for _,(data, target) in enumerate(final_data["train_loader"])]
    test_standard = [(data, target) for _,(data, target) in enumerate(final_data["test_loader"])]

    train_standard_small = [(data, target) for _,(data, target) in enumerate(final_data["train_loader_small"])]
    test_standard_small = [(data, target) for _,(data, target) in enumerate(final_data["test_loader_small"])]

    train_standard_big = [(data, target) for _,(data, target) in enumerate(final_data["train_loader_big"])]
    test_standard_big = [(data, target) for _,(data, target) in enumerate(final_data["test_loader_big"])]

    # training and testing baseline
    my_index_train = 0
    my_index_test = 0
    for w in range(num_workers):
        # train
        data_train = train_standard[my_index_train][0]
        target_train = train_standard[my_index_train][1]
        # test
        data_test = test_standard[my_index_test][0]
        target_test = test_standard[my_index_test][1]
        workers_no_augmented[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
 
        my_index_train+=1
        my_index_test+=1
        if my_index_train > len(train_standard) - 1:
            my_index_train = 0
        if my_index_test > len(test_standard) - 1:
            my_index_test = 0

    # no augmented nodes on small datasets [training and testing]
    my_index_train, my_index_test = 0, 0
    for w in no_augmented_nodes:
        data_train = train_standard_small[my_index_train][0]
        target_train = train_standard_small[my_index_train][1]
        data_test = test_standard_small[my_index_test][0]
        target_test = test_standard_small[my_index_test][1]
        workers_yes_augmented_no_filter[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
        workers_yes_augmented_trustfed[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
        workers_yes_augmented_truflass[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
        my_index_train+=1
        my_index_test+=1
        if my_index_train > len(train_standard_small) - 1:
            my_index_train = 0
        if my_index_test > len(test_standard_small) - 1:
            my_index_test = 0

    # augmented nodes on big datasets [training and testing]
    my_index_train, my_index_test = 0, 0
    for w in augmented_nodes:
        data_train = train_standard_big[my_index_train][0]
        target_train = train_standard_big[my_index_train][1]
        data_test = test_standard_big[my_index_test][0]
        target_test = test_standard_big[my_index_test][1]
        workers_yes_augmented_no_filter[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
        workers_yes_augmented_trustfed[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
        workers_yes_augmented_truflass[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (data_test, target_test))
        my_index_train+=1
        my_index_test+=1
        if my_index_train > len(train_standard_big) - 1:
            my_index_train = 0
        if my_index_test > len(test_standard_big) - 1:
            my_index_test = 0

    # validator nodes
    my_index_test = 0
    for v in range(n_validators):
        data_test = test_standard_big[my_index_test][0]
        target_test = test_standard_big[my_index_test][1]
        validator_workers[v] = Worker(v, 0.1, copy.deepcopy(model_original), (None, None), (data_test, target_test))
        my_index_test+=1
        if my_index_test > len(test_standard_big) - 1:
            my_index_test = 0

    # tester node 
    test_index = random.sample(range(len(test_standard)), 1)[0]
    test_worker = Worker(0, 0.1, copy.deepcopy(model_original), (None, None), (test_standard[test_index][0], test_standard[test_index][1]))

    results_trustfed = np.zeros((num_workers, num_workers))
    # results_truflass = np.zeros((num_workers, num_workers))
    results_truflass = np.zeros((n_validators, num_workers))
    acc_performance_metrics = np.zeros((4,iteration))

    csv_results = [["Iteration", 
                    "no_special_loss", "no_special_f1", "no_special_accuracy",
                    "no_filter_loss", "no_filter_f1", "no_filter_accuracy",
                    "trustfed_loss", "trustfed_f1", "trustfed_accuracy",
                    "truflass_loss", "truflass_f1", "truflass_accuracy"]]

    # ------------------------------------------- learning starts here
    for itr in range(iteration):

        print(f'--------------------- Iteration {itr} ---------------------')
        # train nodes without specials
        for node in workers_no_augmented:
            for _ in range(rounds):
                loss = workers_no_augmented[node].train_my_model()

        # train nodes with augmented but no filter
        for node in workers_yes_augmented_no_filter:
            for _ in range(rounds):
                loss = workers_yes_augmented_no_filter[node].train_my_model()

        # train nodes with augmented and TRUTFED filtering 
        for node in workers_yes_augmented_trustfed:
            for _ in range(rounds):
                loss = workers_yes_augmented_trustfed[node].train_my_model()

        # train nodes with augmented and TRUFLASS filtering 
        for node in workers_yes_augmented_truflass:
            for _ in range(rounds):
                loss = workers_yes_augmented_truflass[node].train_my_model()

        # node democratic validation trustfed
        start_time = time.time()
        for master in workers_yes_augmented_trustfed:
            for slave in workers_yes_augmented_trustfed:
                if int(master/size) == int(slave/size):
                    loss_trustfed = workers_yes_augmented_trustfed[master].test_other_model(workers_yes_augmented_trustfed[slave].id, copy.deepcopy(workers_yes_augmented_trustfed[slave].model), results_trustfed)
                    # loss_trustfed = workers_yes_augmented_truflass[master].test_other_model(workers_yes_augmented_truflass[slave].id, copy.deepcopy(workers_yes_augmented_truflass[slave].model), results_truflass)

        # node democratic validation truflaas - ORACOLO                                                                                                                                                                                       
        for validator in validator_workers:
            for w in all_trainers:
                if int(validator/size) == int(w/size):
                    # loss_trustfed = validator_workers[validator].test_other_model(workers_yes_augmented_trustfed[w].id, copy.deepcopy(workers_yes_augmented_trustfed[w].model), results_trustfed)
                    loss_truflass = validator_workers[validator].test_other_model(workers_yes_augmented_truflass[w].id, copy.deepcopy(workers_yes_augmented_truflass[w].model), results_truflass)

        complaints_trustfed = np.zeros((num_workers))
        complaints_truflass = np.zeros((num_workers))
        malicious_detected_trustfed = []
        malicious_detected_truflass = []

        # cheat = random.sample(no_augmented_nodes, int(len(augmented_nodes)*0.3) )

        start = 0
        end = 0
        for idx, result in enumerate(results_trustfed):
            start = int(idx/size)*size
            end = start + size
            report_trustfed = select_node_to_discard_trustfed(result[start:end])
            for index in report_trustfed:
                complaints_trustfed[index+start] += 1

        start = 0
        end = 0
        for idx, result in enumerate(results_truflass):
            start = int(idx/size)*size
            end = start + size
            report_truflass = select_node_to_discard_truflass(result[start:end])
            for index in report_truflass:
                complaints_truflass[index+start] += 1
            
        for idx, num_complain in enumerate(complaints_trustfed):
            if num_complain >= int(size/2):
                malicious_detected_trustfed += [idx]

        for idx, num_complain in enumerate(complaints_truflass):
            if num_complain >= int(size/2):
                malicious_detected_truflass += [idx]

        print("malicious_detected_trustfed", malicious_detected_trustfed)
        print("malicious_detected_truflass", malicious_detected_truflass)

        # malicious_detected_trustfed = random.sample(malicious_detected_trustfed, int(len(malicious_detected_trustfed)/2))

        end_time = time.time()
        time_taken = end_time - start_time
        print(f'Time taken for validation is {time_taken}')

        models_without_outliers = [workers_no_augmented[index].model for index in workers_no_augmented]
        models_with_outliers_no_filter = [workers_yes_augmented_no_filter[index].model for index in workers_yes_augmented_no_filter]
        models_with_outliers_trustfed_filter = []
        models_with_rares_truflass_filter = []

        for w in workers_yes_augmented_trustfed:
            if not w in malicious_detected_trustfed:
                models_with_outliers_trustfed_filter += [workers_yes_augmented_trustfed[w].model]
            if not w in malicious_detected_truflass:
                models_with_rares_truflass_filter += [workers_yes_augmented_truflass[w].model]

        model_without_outliers_aggreated = aggregate_model(models_without_outliers)
        model_with_outliers_no_filter_aggregated = aggregate_model(models_with_outliers_no_filter)
        model_with_outliers_trustfed_filter_aggregated = aggregate_model(models_with_outliers_trustfed_filter)
        model_with_rare_truflass_filter_aggregated = aggregate_model(models_with_rares_truflass_filter)

        save_np_to_file(f'./results/exp1_augmented_data/{n_augmented}_forgers/models/', f"test_{experiment_counter}_model_without_outliers_aggreated.npy", model_without_outliers_aggreated)
        save_np_to_file(f'./results/exp1_augmented_data/{n_augmented}_forgers/models/', f"test_{experiment_counter}_model_with_outliers_no_filter_aggregated.npy", model_with_outliers_no_filter_aggregated)
        save_np_to_file(f'./results/exp1_augmented_data/{n_augmented}_forgers/models/', f"test_{experiment_counter}_model_with_outliers_trustfed_filter_aggregated.npy", model_with_outliers_trustfed_filter_aggregated)
        save_np_to_file(f'./results/exp1_augmented_data/{n_augmented}_forgers/models/', f"test_{experiment_counter}_model_with_rare_truflass_filter_aggregated.npy", model_with_rare_truflass_filter_aggregated)

        model_without_outliers = Net()
        for i, param in enumerate(model_without_outliers.parameters()):
            param.data = torch.from_numpy(model_without_outliers_aggreated[i]).type('torch.FloatTensor')

        model_with_outliers_no_filter = Net()
        for i, param in enumerate(model_with_outliers_no_filter.parameters()):
            param.data = torch.from_numpy(model_with_outliers_no_filter_aggregated[i]).type('torch.FloatTensor')

        model_with_outliers_trustfed_filter = Net()
        for i, param in enumerate(model_with_outliers_trustfed_filter.parameters()):
            param.data = torch.from_numpy(model_with_outliers_trustfed_filter_aggregated[i]).type('torch.FloatTensor')

        model_with_rare_truflass_filter = Net()
        for i, param in enumerate(model_with_rare_truflass_filter.parameters()):
            param.data = torch.from_numpy(model_with_rare_truflass_filter_aggregated[i]).type('torch.FloatTensor')

        # set aggregated weights on workers
        for w in workers_no_augmented:
            workers_no_augmented[w].set_weights(model_without_outliers_aggreated)
        for w in workers_yes_augmented_no_filter:
            workers_yes_augmented_no_filter[w].set_weights(model_with_outliers_no_filter_aggregated)
        for w in workers_yes_augmented_trustfed:
            workers_yes_augmented_trustfed[w].set_weights(model_with_outliers_trustfed_filter_aggregated)
        for w in workers_yes_augmented_truflass:
            workers_yes_augmented_truflass[w].set_weights(model_with_rare_truflass_filter_aggregated)

        model_performance_without_outliers = test_worker.test_final_model(model_without_outliers)
        model_performance_with_outliers_no_filter = test_worker.test_final_model(model_with_outliers_no_filter)
        model_performance_with_outliers_trustfed_filter = test_worker.test_final_model(model_with_outliers_trustfed_filter)
        model_performance_with_rares_truflass_filter = test_worker.test_final_model(model_with_rare_truflass_filter)
        print(f'Loss no outliers is {model_performance_without_outliers["loss"]}')
        print(f'Loss with outliers no filter {model_performance_with_outliers_no_filter["loss"]}')
        print(f'Loss with outliers trustfed filter {model_performance_with_outliers_trustfed_filter["loss"]}')
        print(f'Loss with rares truflass filter {model_performance_with_rares_truflass_filter["loss"]}')
        print("----------------------------------")
        print(f'acc no outliers is {model_performance_without_outliers["accuracy"]}')
        print(f'acc with outliers no filter {model_performance_with_outliers_no_filter["accuracy"]}')
        print(f'acc with outliers trustfed filter {model_performance_with_outliers_trustfed_filter["accuracy"]}')
        print(f'acc with rares truflass filter {model_performance_with_rares_truflass_filter["accuracy"]}')

        acc_performance_metrics[0][itr] = model_performance_without_outliers["accuracy"]
        acc_performance_metrics[1][itr] = model_performance_with_outliers_no_filter["accuracy"]
        acc_performance_metrics[2][itr] = model_performance_with_outliers_trustfed_filter["accuracy"]
        acc_performance_metrics[3][itr] = model_performance_with_rares_truflass_filter["accuracy"]

        this_run_results = []
        this_run_results.append(itr)
        this_run_results.append(model_performance_without_outliers["loss"])
        this_run_results.append(model_performance_without_outliers["f1"])
        this_run_results.append(model_performance_without_outliers["accuracy"])

        this_run_results.append(model_performance_with_outliers_no_filter["loss"])
        this_run_results.append(model_performance_with_outliers_no_filter["f1"])
        this_run_results.append(model_performance_with_outliers_no_filter["accuracy"])

        this_run_results.append(model_performance_with_outliers_trustfed_filter["loss"])
        this_run_results.append(model_performance_with_outliers_trustfed_filter["f1"])
        this_run_results.append(model_performance_with_outliers_trustfed_filter["accuracy"])

        this_run_results.append(model_performance_with_rares_truflass_filter["loss"])
        this_run_results.append(model_performance_with_rares_truflass_filter["f1"])
        this_run_results.append(model_performance_with_rares_truflass_filter["accuracy"])
        csv_results.append(this_run_results)

        save_2d_matrix_to_csv_file(f'./results/exp1_augmented_data/{n_augmented}_forgers/csvs/', f"test_{experiment_counter}.csv", csv_results)

    print("acc_performance_metrics[1]", acc_performance_metrics[1])
    print("acc_performance_metrics[2]", acc_performance_metrics[2])
    print("acc_performance_metrics[3]", acc_performance_metrics[3])

    plt.figure(figsize = (5,5))
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy [%]')
    plt.ylim([60, 90])
    plt.plot(range(iteration), acc_performance_metrics[1]*100, label='No node filtering', linestyle = ':')
    plt.plot(range(iteration), acc_performance_metrics[2]*100, label='TrustFed', linestyle = '--')
    plt.plot(range(iteration), acc_performance_metrics[3]*100, label='TruFLaaS', color="red", linestyle = '-')
    plt.axhline(y = 80, color = 'gray', linestyle = '--')
    plt.locator_params(axis="x", nbins=5)
    plt.legend()
    plt.title(f"Node selection strategy with {n_augmented} nodes with augmented data")
    create_folder_if_not_exists(f'./results/exp1_augmented_data/{n_augmented}_forgers/pngs/')
    create_folder_if_not_exists(f'./results/exp1_augmented_data/{n_augmented}_forgers/pdfs/')
    plt.savefig(f'./results/exp1_augmented_data/{n_augmented}_forgers/pngs/test_{experiment_counter}.png', format="png", bbox_inches='tight')
    plt.savefig(f'./results/exp1_augmented_data/{n_augmented}_forgers/pdfs/test_{experiment_counter}.pdf', format="pdf", bbox_inches='tight')

if __name__ == '__main__':
    final_data = process_data_final()
    
    for n_specials in special_options:
        for experiment_counter in range(experiments):
            print("---------------- REVIEW -------------------")
            print("running experiment:", experiment_counter)
            print("-------------------------------------------")
            experiment_rare_cases(experiment_counter, n_specials, final_data)
    play('./data/alarm.mp3')
    


# ------------------------------- TO DO ---------------------------------------------------------------------

# 1. tutti i nodi vanno bene
# 2. tutti i nodi malevoli (malevoli inclusi)
# 3. detection dei nodi malevoli con il nostro algoritmo
# 4. confronto con il loro approccio


# graph A: 1,2,3
# graph B: 3,4
# graph C: confronto tempi di inferenza tra la nostra soluzione e trustFed
# graph D: cercare i casi rari, o nei tempi di rottura o nei valori dentro al training

# -----------------------------------------------------------------------------------------------------------

# quanti nodi riusciamo a detect noi e quanti nodi riescono a fare detection loro
# metriche di misurazione dell'accuratezza diverse (f1, MAPE, R2)
# trustfed non prioritizza i nodi con loss piu' bassa

# fare forgiatura e far vedere come il loro fa schifo
# mantenere quindi una parte del dataset per fare il testing finale, il loro modello deve fare schifo
# il nostro deve passare

# trustfullness del nodo equivale al peso di aggregazione

# last but not least
# risoluzione caso raro: quelli che hanno RUL < 10
# il loro non deve passare perche' vengono aggregati anche i nodi che 
# non riescono ad identificare i casi rari
# mentre la nostra soluzione ci deve riuscire