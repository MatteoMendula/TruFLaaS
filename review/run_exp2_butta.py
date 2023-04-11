import utils
import constants
import random
from threading import Thread
import custom_extension

import tensorflow as tf
import keras
import numpy as np

DEBUG = True

def run_single_case(
        experiment_name, 
        
        X_train, 
        y_train,
        X_train_no_rare,
        y_train_no_rare,
        
        how_small_percentage,
        percentage_no_rares_clients, 
        input_shape, 
        nb_classes, 
        class_weights,

        test_batched_overall, 
        test_batched_reduced,
        test_batched_reduced_only_rare):
    
    # to run faster every client is how_small_percentage times smaller
    percentage_small_clients = 1
    
    print("Creating clients...")
    clients_batches_original = utils.create_clients(X_train, y_train, nb_classes, constants.sampling_technique, num_clients=constants.num_clients, initial='client')
    clients_batches_no_rares = utils.create_clients(X_train_no_rare, y_train_no_rare, nb_classes, constants.sampling_technique, num_clients=constants.num_clients, initial='client')
    client_names = list(clients_batches_original.keys())
    random.shuffle(client_names)

    # this is meant to give to all clients the same samples indexes
    original_sample_size = len(clients_batches_original[client_names[0]])
    small_sample_size = int(original_sample_size * how_small_percentage)
    sample_indices = np.random.choice(original_sample_size, small_sample_size, replace=False)

    # create small batches
    clients_batched_dic = {}
    clients_batched_dic["NO_SELECTION"] = custom_extension.create_noisy_batches(clients_batches_original, clients_batches_no_rares, percentage_no_rares_clients, sample_indices)
    clients_batched_dic["UNION"] = custom_extension.create_noisy_batches(clients_batches_original, clients_batches_no_rares, percentage_no_rares_clients, sample_indices)
    clients_batched_dic["INTERSECTION"] = custom_extension.create_noisy_batches(clients_batches_original, clients_batches_no_rares, percentage_no_rares_clients, sample_indices)

    global_model = {}
    global_model["NO_SELECTION"] : keras.Model = utils.get_model(input_shape, nb_classes)
    global_model["UNION"] : keras.Model = utils.get_model(input_shape, nb_classes)
    global_model["INTERSECTION"] : keras.Model = utils.get_model(input_shape, nb_classes)

    # keep track of best global accuracy
    best_accuracy_overall : dict = {}
    best_accuracy_overall["NO_SELECTION"] : float = 0
    best_accuracy_overall["UNION"] : float = 0
    best_accuracy_overall["INTERSECTION"] : float = 0

    if DEBUG:
        for client_name in client_names:
            print("{}_no_selection".format(client_name), len(clients_batched_dic["NO_SELECTION"][client_name]))
            print("{}UNION".format(client_name), len(clients_batched_dic["UNION"][client_name]))
            print("{}INTERSECTION".format(client_name), len(clients_batched_dic["INTERSECTION"][client_name]))

    client_set : dict = {}
    client_set["NO_SELECTION"] : dict = {k: {} for k in client_names}
    client_set["UNION"] : dict = {k: {} for k in client_names}
    client_set["INTERSECTION"] : dict = {k: {} for k in client_names}

    for (client_name, data) in clients_batched_dic["NO_SELECTION"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss, 
                        optimizer=constants.optimizer, 
                        metrics=constants.metrics)
        
        client_set["NO_SELECTION"][client_name]["model"] = local_model
        client_set["NO_SELECTION"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)

    for (client_name, data) in clients_batched_dic["UNION"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss, 
                        optimizer=constants.optimizer, 
                        metrics=constants.metrics)
        
        client_set["UNION"][client_name]["model"] = local_model
        client_set["UNION"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)

    for (client_name, data) in clients_batched_dic["INTERSECTION"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss, 
                        optimizer=constants.optimizer, 
                        metrics=constants.metrics)
        
        client_set["INTERSECTION"][client_name]["model"] = local_model
        client_set["INTERSECTION"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)

    # initialize testing metrics
    testing_metrics : dict = {}
    testing_metrics["NO_SELECTION"] = {}
    testing_metrics["UNION"] = {}
    testing_metrics["INTERSECTION"] = {}

    testing_metrics["NO_SELECTION"]["loss"] = []
    testing_metrics["NO_SELECTION"]["accuracy"] = []
    testing_metrics["NO_SELECTION"]["precision"] = []
    testing_metrics["NO_SELECTION"]["recall"] = []
    testing_metrics["NO_SELECTION"]["f1"] = []
    testing_metrics["NO_SELECTION"]["best_global_accuracy"] = 0

    testing_metrics["UNION"]["loss"] = []
    testing_metrics["UNION"]["accuracy"] = []
    testing_metrics["UNION"]["precision"] = []
    testing_metrics["UNION"]["recall"] = []
    testing_metrics["UNION"]["f1"] = []
    testing_metrics["UNION"]["best_global_accuracy"] = 0

    testing_metrics["INTERSECTION"]["loss"] = []
    testing_metrics["INTERSECTION"]["accuracy"] = []
    testing_metrics["INTERSECTION"]["precision"] = []
    testing_metrics["INTERSECTION"]["recall"] = []
    testing_metrics["INTERSECTION"]["f1"] = []
    testing_metrics["INTERSECTION"]["best_global_accuracy"] = 0

    threads : dict = {}
    threads["NO_SELECTION"] = [None] * constants.num_clients
    threads["UNION"] = [None] * constants.num_clients
    threads["INTERSECTION"] = [None] * constants.num_clients
    
    print("Starting training...")
    for comm_round in range(constants.comms_round):            
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights : dict = {}
        global_weights["NO_SELECTION"] = global_model["NO_SELECTION"].get_weights()
        global_weights["UNION"] = global_model["UNION"].get_weights()
        global_weights["INTERSECTION"] = global_model["INTERSECTION"].get_weights()
        
        #initial list to collect local model weights after scalling
        local_weight_list : dict = {}
        local_weight_list["NO_SELECTION"] : list = list()
        local_weight_list["UNION"] : list = list()
        local_weight_list["INTERSECTION"] : list = list()

        average_weights : dict = {}
        average_weights["NO_SELECTION"] : list = list()
        average_weights["UNION"] : list = list()
        average_weights["INTERSECTION"] : list = list()
        
        #loop through each client and create new local model
        # no_selection_clients
        for i, client_name in enumerate(client_names):    
            threads["NO_SELECTION"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["NO_SELECTION"], class_weights, client_set["NO_SELECTION"], comm_round))     
            threads["NO_SELECTION"][i].start()

        for i in range(constants.num_clients):
            threads["NO_SELECTION"][i].join()         

        # truflaas_selection_clients
        for i, client_name in enumerate(client_names):    
            threads["UNION"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["UNION"], class_weights, client_set["UNION"], comm_round))     
            threads["UNION"][i].start()

        for i in range(constants.num_clients):
            threads["UNION"][i].join()     

        # trustfed_selection_clients
        for i, client_name in enumerate(client_names):
            threads["INTERSECTION"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["INTERSECTION"], class_weights, client_set["INTERSECTION"], comm_round))     
            threads["INTERSECTION"][i].start()
        
        for i in range(constants.num_clients):
            threads["INTERSECTION"][i].join()

        local_weight_list["NO_SELECTION"] : list = custom_extension.select_all_clients(client_set["NO_SELECTION"], test_batched_reduced, comm_round)
        local_weight_list["UNION"] : list = custom_extension.select_best_clients(client_set["UNION"], test_batched_reduced, comm_round, mode = "UNION", test_batch_rares = test_batched_reduced_only_rare)
        local_weight_list["INTERSECTION"] : list = custom_extension.select_best_clients(client_set["INTERSECTION"], test_batched_reduced, comm_round, mode = "INTERSECTION", test_batch_rares = test_batched_reduced_only_rare)

        #to get the average over all the local model, we simply calculate the average of the sum of local weights
        average_weights["NO_SELECTION"] : list = utils.average_model_weights(local_weight_list["NO_SELECTION"])
        average_weights["UNION"] : list = utils.average_model_weights(local_weight_list["UNION"])
        average_weights["INTERSECTION"] : list = utils.average_model_weights(local_weight_list["INTERSECTION"])

        #update global model 
        global_model["NO_SELECTION"].set_weights(average_weights["NO_SELECTION"])
        global_model["UNION"].set_weights(average_weights["UNION"])
        global_model["INTERSECTION"].set_weights(average_weights["INTERSECTION"])

        # testing global model with NO_SELECTION
        for(x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["NO_SELECTION"], comm_round, "global no_selection")
            testing_metrics["NO_SELECTION"]["loss"].append(g_loss)
            testing_metrics["NO_SELECTION"]["accuracy"].append(g_accuracy)
            testing_metrics["NO_SELECTION"]["precision"].append(g_precision)
            testing_metrics["NO_SELECTION"]["recall"].append(g_recall)
            testing_metrics["NO_SELECTION"]["f1"].append(g_f1)
   
            if g_accuracy > best_accuracy_overall["NO_SELECTION"]:
                best_accuracy_overall["NO_SELECTION"] = g_accuracy
                global_model["NO_SELECTION"].save_weights('global_model_best_no_selection.h5')
                print("New NO_SELECTION Weights Saved")
            
        # testing global model with UNION
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["UNION"], comm_round, "global truflaas")
            testing_metrics["UNION"]["loss"].append(g_loss)
            testing_metrics["UNION"]["accuracy"].append(g_accuracy)
            testing_metrics["UNION"]["precision"].append(g_precision)
            testing_metrics["UNION"]["recall"].append(g_recall)
            testing_metrics["UNION"]["f1"].append(g_f1)

            if g_accuracy > best_accuracy_overall["UNION"]:
                best_accuracy_overall["UNION"] = g_accuracy
                global_model["UNION"].save_weights('global_model_best_truflaas.h5')
                print("New UNION Weights Saved")

        # testing global model with INTERSECTION
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["INTERSECTION"], comm_round, "global trustfed")
            testing_metrics["INTERSECTION"]["loss"].append(g_loss)
            testing_metrics["INTERSECTION"]["accuracy"].append(g_accuracy)
            testing_metrics["INTERSECTION"]["precision"].append(g_precision)
            testing_metrics["INTERSECTION"]["recall"].append(g_recall)
            testing_metrics["INTERSECTION"]["f1"].append(g_f1)

            if g_accuracy > best_accuracy_overall["INTERSECTION"]:
                best_accuracy_overall["INTERSECTION"] = g_accuracy
                global_model["INTERSECTION"].save_weights('global_model_best_trustfed.h5')
                print("New INTERSECTION Weights Saved")

    print("Best Accuracy Overall NO_SELECTION: ", best_accuracy_overall["NO_SELECTION"])
    print("Best Accuracy Overall UNION: ", best_accuracy_overall["UNION"])
    print("Best Accuracy Overall INTERSECTION: ", best_accuracy_overall["INTERSECTION"])

    for round in range(constants.comms_round):
        print("Round: ", round)
        print("NO_SELECTION: ", testing_metrics["NO_SELECTION"]["accuracy"][round])
        print("UNION: ", testing_metrics["UNION"]["accuracy"][round])
        print("INTERSECTION: ", testing_metrics["INTERSECTION"]["accuracy"][round])

    print("-------------------- --------------------- ---------------------")

    print(testing_metrics["NO_SELECTION"])
    print(testing_metrics["UNION"])
    print(testing_metrics["INTERSECTION"])

    print("-------------------- --------------------- ---------------------")
    print("saving and showing graphs")

    custom_extension.save_graphs(testing_metrics, experiment_name, percentage_small_clients)

    print("-------------------- --------------------- ---------------------")
    print("saving csv file")

    custom_extension.save_csv(testing_metrics, experiment_name, percentage_small_clients)


if __name__ == "__main__":
    #initialize global model

    # experiments 2 --------------------------------------- 
    experiment_name = "exp2"
    how_small_percentage = 0.01
    runs = [
        {
            "percentage_no_rares_clients": 0
        },
        {
            "percentage_no_rares_clients": 0.1
        },
        {
            "percentage_no_rares_clients": 0.25
        }
    ]

    df = utils.load_data()
    df_no_rare = custom_extension.remove_rare_cases_from_df(df)
    df_only_rare = custom_extension.get_rare_cases_from_df(df)
    X_train, y_train, X_test, y_test, label_encoder = utils.split_df(df)
    X_train_no_rare, y_train_no_rare, X_test_no_rare, y_test_no_rare, label_encoder_no_rare = utils.split_df(df_no_rare)
    X_train_only_rare, y_train_only_rare, X_test_only_rare, y_test_only_rare, label_encoder_only_rare = utils.split_df(df_only_rare)

    input_shape = X_train.shape[1:]
    nb_classes = len(label_encoder.classes_)
    class_weights = utils.get_class_weights(y_train)

    y_test = utils.convert_to_categorical(y_test, nb_classes)
    y_test_only_rare = utils.convert_to_categorical(y_test_only_rare, nb_classes)

    # reduce test set size ----------------------------------------------------- LOOK HERE
    X_test_reduced, y_test_reduced = custom_extension.sample_test(X_test, y_test, 0.2)
    X_test_reduced_only_rare, y_test_reduced_only_rare = custom_extension.sample_test(X_test_only_rare, y_test_only_rare, 0.2)

    test_batched_overall = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    test_batched_reduced = tf.data.Dataset.from_tensor_slices((X_test_reduced, y_test_reduced)).batch(len(y_test_reduced))
    test_batched_reduced_only_rare = tf.data.Dataset.from_tensor_slices((X_test_reduced_only_rare, y_test_reduced_only_rare)).batch(len(y_test_reduced_only_rare))

    for r in runs:
        _percentage_no_rares_clients = r["percentage_no_rares_clients"]
        print("percentage_no_rares_clients: ", _percentage_no_rares_clients)
        print("experiment_name: ", experiment_name)
        print("how_small_percentage: ", how_small_percentage)
        run_single_case(experiment_name = experiment_name,
            
                        X_train = X_train, 
                        y_train = y_train, 

                        X_train_no_rare = X_train_no_rare,
                        y_train_no_rare = y_train_no_rare,

                        how_small_percentage = how_small_percentage,
                        percentage_no_rares_clients = _percentage_no_rares_clients, 
                        input_shape=input_shape, 
                        nb_classes=nb_classes, 
                        class_weights=class_weights, 

                        test_batched_overall=test_batched_overall, 
                        test_batched_reduced=test_batched_reduced,
                        test_batched_reduced_only_rare=test_batched_reduced_only_rare)