import utils
import constants
import random
from threading import Thread
import custom_extension

import tensorflow as tf
import keras
import numpy as np
from playsound import playsound as play


DEBUG = False

def run_single_case(
        experiment_name, 
        client_names,

        X_train, 
        y_train, 
        X_test,
        y_test,

        X_train_no_rare, 
        y_train_no_rare, 
        X_test_no_rare, 
        y_test_no_rare,
        
        percentage_no_rares_clients,
        how_small_percentage,
        special_clients, 
        input_shape, 
        nb_classes, 
        class_weights,
        test_batched_overall,
        test_batched_truflass,
        test_batched_truflass_only_rare):
    
    print("Creating clients...")
    clients_batched_original = utils.create_clients(X_train, y_train, nb_classes, constants.sampling_technique, client_names)    
    clients_batched_no_rares = utils.create_clients(X_train_no_rare, y_train_no_rare, nb_classes, constants.sampling_technique, client_names)    
    # clients_batched_only_rares = utils.create_clients(X_train_only_rare, y_train_only_rare, nb_classes, constants.sampling_technique, client_names)    
    client_names = list(clients_batched_original.keys())

    # this is meant to give to all clients the same samples indexes
    original_sample_size = len(clients_batched_original[client_names[0]])
    small_sample_size = int(original_sample_size * how_small_percentage)
    sample_indices = np.random.choice(original_sample_size, small_sample_size, replace=False)

    # original_sample_size 22087
    # small_sample_size 220
    # len sample_indices 220

    print("original_sample_size", original_sample_size)
    print("small_sample_size", small_sample_size)
    print("len sample_indices", len(sample_indices))

    print("sample_indices", sample_indices)

    global_model = {}
    global_model["NO_SELECTION"] : keras.Model = utils.get_model(input_shape, nb_classes)
    global_model["UNION"] : keras.Model = utils.get_model(input_shape, nb_classes)
    global_model["INTERSECTION"] : keras.Model = utils.get_model(input_shape, nb_classes)
    global_model["OVERALL"] : keras.Model = utils.get_model(input_shape, nb_classes)
    global_model["RARES"] : keras.Model = utils.get_model(input_shape, nb_classes)
    
    # create small batches
    clients_batched_dic = {}
    clients_batched_dic["NO_SELECTION"] = custom_extension.create_batches_with_no_rares(clients_batched_original, clients_batched_no_rares, special_clients, sample_indices)
    clients_batched_dic["UNION"] = custom_extension.create_batches_with_no_rares(clients_batched_original, clients_batched_no_rares, special_clients, sample_indices)
    clients_batched_dic["INTERSECTION"] = custom_extension.create_batches_with_no_rares(clients_batched_original, clients_batched_no_rares, special_clients, sample_indices)
    clients_batched_dic["OVERALL"] = custom_extension.create_batches_with_no_rares(clients_batched_original, clients_batched_no_rares, special_clients, sample_indices)
    clients_batched_dic["RARES"] = custom_extension.create_batches_with_no_rares(clients_batched_original, clients_batched_no_rares, special_clients, sample_indices)


    # keep track of best global accuracy
    # best_accuracy_overall : dict = {}
    # best_accuracy_overall["NO_SELECTION"] : float = 0
    # best_accuracy_overall["UNION"] : float = 0
    # best_accuracy_overall["INTERSECTION"] : float = 0

    if DEBUG:
        for client_name in client_names:
            print(" - - - TRAIN - - -")
            print("{}_no_selection".format(client_name), len(clients_batched_dic["NO_SELECTION"][client_name]))
            print("{}UNION".format(client_name), len(clients_batched_dic["UNION"][client_name]))
            print("{}INTERSECTION".format(client_name), len(clients_batched_dic["INTERSECTION"][client_name]))
            print("{}OVERALL".format(client_name), len(clients_batched_dic["OVERALL"][client_name]))
            print("{}RARES".format(client_name), len(clients_batched_dic["RARES"][client_name]))

    client_set : dict = {}
    client_set["NO_SELECTION"] : dict = {k: {} for k in client_names}
    client_set["UNION"] : dict = {k: {} for k in client_names}
    client_set["INTERSECTION"] : dict = {k: {} for k in client_names}
    client_set["OVERALL"] : dict = {k: {} for k in client_names}
    client_set["RARES"] : dict = {k: {} for k in client_names}

    for (client_name, data) in clients_batched_dic["NO_SELECTION"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss,
                        optimizer=constants.optimizer,
                        metrics=constants.metrics)
        
        # creating test samples for TRUEFLAAS and INTERSECTION too
        if client_name in special_clients:
            X_test_sample, y_test_sample = custom_extension.sample_test(X_test_no_rare, y_test_no_rare, constants.local_testing_size)
        else:
            X_test_sample, y_test_sample = custom_extension.sample_test(X_test, y_test, 1)

        client_set["NO_SELECTION"][client_name]["model"] = local_model
        client_set["NO_SELECTION"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)
        client_set["NO_SELECTION"][client_name]["testing"] = custom_extension.create_testing_batched(X_test_sample, y_test_sample)
        

    for (client_name, data) in clients_batched_dic["UNION"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss, 
                        optimizer=constants.optimizer, 
                        metrics=constants.metrics)
    
        client_set["UNION"][client_name]["model"] = local_model
        client_set["UNION"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)
        # client_set["UNION"][client_name]["testing"] = custom_extension.create_testing_batched(client_batches_testing_no_selection[client_name]["X_test_sample"], client_batches_testing_no_selection[client_name]["y_test_sample"])

    for (client_name, data) in clients_batched_dic["INTERSECTION"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss, 
                        optimizer=constants.optimizer, 
                        metrics=constants.metrics)
        client_set["INTERSECTION"][client_name]["model"] = local_model
        client_set["INTERSECTION"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)
        # client_set["INTERSECTION"][client_name]["testing"] = custom_extension.create_testing_batched(client_batches_testing_no_selection[client_name]["X_test_sample"], client_batches_testing_no_selection[client_name]["y_test_sample"])
    
    for (client_name, data) in clients_batched_dic["OVERALL"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss,
                        optimizer=constants.optimizer,
                        metrics=constants.metrics)
        client_set["OVERALL"][client_name]["model"] = local_model
        client_set["OVERALL"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)

    for (client_name, data) in clients_batched_dic["RARES"].items():
        local_model = utils.get_model(input_shape, nb_classes)
        local_model.compile(loss=constants.loss,
                        optimizer=constants.optimizer,
                        metrics=constants.metrics)
        client_set["RARES"][client_name]["model"] = local_model
        client_set["RARES"][client_name]["dataset"] = utils.batch_data(data, constants.BATCH_SIZE)
        
    # initialize testing metrics
    testing_metrics : dict = {}
    testing_metrics["NO_SELECTION"] = {}
    testing_metrics["UNION"] = {}
    testing_metrics["INTERSECTION"] = {}
    testing_metrics["OVERALL"] = {}
    testing_metrics["RARES"] = {}

    testing_metrics["NO_SELECTION"]["loss"] = []
    testing_metrics["NO_SELECTION"]["accuracy"] = []
    testing_metrics["NO_SELECTION"]["precision"] = []
    testing_metrics["NO_SELECTION"]["recall"] = []
    testing_metrics["NO_SELECTION"]["f1"] = []

    testing_metrics["UNION"]["loss"] = []
    testing_metrics["UNION"]["accuracy"] = []
    testing_metrics["UNION"]["precision"] = []
    testing_metrics["UNION"]["recall"] = []
    testing_metrics["UNION"]["f1"] = []

    testing_metrics["INTERSECTION"]["loss"] = []
    testing_metrics["INTERSECTION"]["accuracy"] = []
    testing_metrics["INTERSECTION"]["precision"] = []
    testing_metrics["INTERSECTION"]["recall"] = []
    testing_metrics["INTERSECTION"]["f1"] = []

    testing_metrics["OVERALL"]["loss"] = []
    testing_metrics["OVERALL"]["accuracy"] = []
    testing_metrics["OVERALL"]["precision"] = []
    testing_metrics["OVERALL"]["recall"] = []
    testing_metrics["OVERALL"]["f1"] = []

    testing_metrics["RARES"]["loss"] = []
    testing_metrics["RARES"]["accuracy"] = []
    testing_metrics["RARES"]["precision"] = []
    testing_metrics["RARES"]["recall"] = []
    testing_metrics["RARES"]["f1"] = []

    threads : dict = {}
    threads["NO_SELECTION"] = [None] * constants.num_clients
    threads["UNION"] = [None] * constants.num_clients
    threads["INTERSECTION"] = [None] * constants.num_clients
    threads["OVERALL"] = [None] * constants.num_clients
    threads["RARES"] = [None] * constants.num_clients
    
    print("Starting training...")
    for comm_round in range(constants.comms_round):            
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights : dict = {}
        global_weights["NO_SELECTION"] = global_model["NO_SELECTION"].get_weights()
        global_weights["UNION"] = global_model["UNION"].get_weights()
        global_weights["INTERSECTION"] = global_model["INTERSECTION"].get_weights()
        global_weights["OVERALL"] = global_model["OVERALL"].get_weights()
        global_weights["RARES"] = global_model["RARES"].get_weights()
        
        #initial list to collect local model weights after scalling
        local_weight_list : dict = {}
        local_weight_list["NO_SELECTION"] : list = list()
        local_weight_list["UNION"] : list = list()
        local_weight_list["INTERSECTION"] : list = list()
        local_weight_list["OVERALL"] : list = list()
        local_weight_list["RARES"] : list = list()

        average_weights : dict = {}
        average_weights["NO_SELECTION"] : list = list()
        average_weights["UNION"] : list = list()
        average_weights["INTERSECTION"] : list = list()
        average_weights["OVERALL"] : list = list()
        average_weights["RARES"] : list = list()

        # create appropriate testing batches UNION
        this_round_truflass_testing = test_batched_truflass[comm_round]
        test_batched_truflaas_overall = custom_extension.create_testing_batched(this_round_truflass_testing["x"], this_round_truflass_testing["y"])
        # create appropriate testing batches UNION rare
        this_round_truflass_testing_rare = test_batched_truflass_only_rare[comm_round]
        test_batched_truflaas_rare = custom_extension.create_testing_batched(this_round_truflass_testing_rare["x"], this_round_truflass_testing_rare["y"])

        #loop through each client and create new local model
        # no_selection_clients
        for i, client_name in enumerate(client_names):    
            threads["NO_SELECTION"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["NO_SELECTION"], class_weights, client_set["NO_SELECTION"], comm_round))     
            threads["NO_SELECTION"][i].start()

        for i in range(constants.num_clients):
            threads["NO_SELECTION"][i].join()         

        # union clients
        for i, client_name in enumerate(client_names):    
            threads["UNION"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["UNION"], class_weights, client_set["UNION"], comm_round))     
            threads["UNION"][i].start()

        for i in range(constants.num_clients):
            threads["UNION"][i].join()     

        # intersection clients
        for i, client_name in enumerate(client_names):
            threads["INTERSECTION"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["INTERSECTION"], class_weights, client_set["INTERSECTION"], comm_round))     
            threads["INTERSECTION"][i].start()
        
        for i in range(constants.num_clients):
            threads["INTERSECTION"][i].join()

        # overall clients
        for i, client_name in enumerate(client_names):
            threads["OVERALL"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["OVERALL"], class_weights, client_set["OVERALL"], comm_round))     
            threads["OVERALL"][i].start()

        for i in range(constants.num_clients):
            threads["OVERALL"][i].join()

        # rare clients
        for i, client_name in enumerate(client_names):
            threads["RARES"][i] : Thread = Thread(target=utils.train_client, args=(client_name, global_weights["RARES"], class_weights, client_set["RARES"], comm_round))     
            threads["RARES"][i].start()

        for i in range(constants.num_clients):
            threads["RARES"][i].join()

        local_weight_list["NO_SELECTION"] : list = custom_extension.select_all_clients(client_set["NO_SELECTION"])
        local_weight_list["UNION"] : list = custom_extension.select_best_clients_exp2(client_set["UNION"], test_batched_truflaas_overall, test_batched_truflaas_rare, comm_round, mode = "UNION", experiment_name = experiment_name)
        local_weight_list["INTERSECTION"] : list = custom_extension.select_best_clients_exp2(client_set["INTERSECTION"], test_batched_truflaas_overall, test_batched_truflaas_rare, comm_round, mode = "INTERSECTION", experiment_name = experiment_name)
        local_weight_list["OVERALL"] : list = custom_extension.select_best_clients_exp2(client_set["OVERALL"], test_batched_truflaas_overall, test_batched_truflaas_rare, comm_round, mode = "OVERALL", experiment_name = experiment_name)
        local_weight_list["RARES"] : list = custom_extension.select_best_clients_exp2(client_set["RARES"], test_batched_truflaas_overall, test_batched_truflaas_rare, comm_round, mode = "RARES", experiment_name = experiment_name)

        #to get the average over all the local model, we simply calculate the average of the sum of local weights
        average_weights["NO_SELECTION"] : list = utils.average_model_weights(local_weight_list["NO_SELECTION"])
        average_weights["UNION"] : list = utils.average_model_weights(local_weight_list["UNION"])
        average_weights["INTERSECTION"] : list = utils.average_model_weights(local_weight_list["INTERSECTION"])
        average_weights["OVERALL"] : list = utils.average_model_weights(local_weight_list["OVERALL"])
        average_weights["RARES"] : list = utils.average_model_weights(local_weight_list["RARES"])

        #update global model 
        global_model["NO_SELECTION"].set_weights(average_weights["NO_SELECTION"])
        global_model["UNION"].set_weights(average_weights["UNION"])
        global_model["INTERSECTION"].set_weights(average_weights["INTERSECTION"])
        global_model["OVERALL"].set_weights(average_weights["OVERALL"])
        global_model["RARES"].set_weights(average_weights["RARES"])

        # testing global model with NO_SELECTION
        for(x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["NO_SELECTION"], comm_round, "[{}] global NO_SELECTION".format(experiment_name))
            testing_metrics["NO_SELECTION"]["loss"].append(g_loss)
            testing_metrics["NO_SELECTION"]["accuracy"].append(g_accuracy)
            testing_metrics["NO_SELECTION"]["precision"].append(g_precision)
            testing_metrics["NO_SELECTION"]["recall"].append(g_recall)
            testing_metrics["NO_SELECTION"]["f1"].append(g_f1)
            
        # testing global model with UNION
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["UNION"], comm_round, "[{}] global UNION".format(experiment_name))
            testing_metrics["UNION"]["loss"].append(g_loss)
            testing_metrics["UNION"]["accuracy"].append(g_accuracy)
            testing_metrics["UNION"]["precision"].append(g_precision)
            testing_metrics["UNION"]["recall"].append(g_recall)
            testing_metrics["UNION"]["f1"].append(g_f1)

        # testing global model with INTERSECTION
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["INTERSECTION"], comm_round, "[{}] global INTERSECTION".format(experiment_name))
            testing_metrics["INTERSECTION"]["loss"].append(g_loss)
            testing_metrics["INTERSECTION"]["accuracy"].append(g_accuracy)
            testing_metrics["INTERSECTION"]["precision"].append(g_precision)
            testing_metrics["INTERSECTION"]["recall"].append(g_recall)
            testing_metrics["INTERSECTION"]["f1"].append(g_f1)

        # testing global model with OVERALL
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["OVERALL"], comm_round, "[{}] global OVERALL".format(experiment_name))
            testing_metrics["OVERALL"]["loss"].append(g_loss)
            testing_metrics["OVERALL"]["accuracy"].append(g_accuracy)
            testing_metrics["OVERALL"]["precision"].append(g_precision)
            testing_metrics["OVERALL"]["recall"].append(g_recall)
            testing_metrics["OVERALL"]["f1"].append(g_f1)
        
        # testing global model with RARES
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["RARES"], comm_round, "[{}] global RARES".format(experiment_name))
            testing_metrics["RARES"]["loss"].append(g_loss)
            testing_metrics["RARES"]["accuracy"].append(g_accuracy)
            testing_metrics["RARES"]["precision"].append(g_precision)
            testing_metrics["RARES"]["recall"].append(g_recall)
            testing_metrics["RARES"]["f1"].append(g_f1)

        # testing global model with RARES
        for (x_batch, y_batch) in test_batched_overall:
            g_loss, g_accuracy, g_precision, g_recall, g_f1 = utils.test_model(x_batch, y_batch, global_model["RARES"], comm_round, "[{}] global RARES".format(experiment_name))
            testing_metrics["RARES"]["loss"].append(g_loss)
            testing_metrics["RARES"]["accuracy"].append(g_accuracy)
            testing_metrics["RARES"]["precision"].append(g_precision)
            testing_metrics["RARES"]["recall"].append(g_recall)
            testing_metrics["RARES"]["f1"].append(g_f1)

    print("-------------------- --------------------- ---------------------")
    print("saving and showing graphs")

    custom_extension.save_graphs(testing_metrics, experiment_name, percentage_no_rares_clients, True)

    print("-------------------- --------------------- ---------------------")
    print("saving csv file")

    custom_extension.save_csv(testing_metrics, experiment_name, percentage_no_rares_clients, )


if __name__ == "__main__":
    #initialize global model

    # experiments 2 --------------------------------------- 
    experiment_name = "exp2"
    how_small_percentage = 0.01
    runs = [
        {
            "percentage_no_rares_clients": 1
        },
        {
            "percentage_no_rares_clients": 0.9
        },
        {
            "percentage_no_rares_clients": 0.75
        }
    ]

    df = utils.load_data()
    df_no_rare = custom_extension.remove_rare_cases_from_df(df)
    df_only_rare = custom_extension.get_rare_cases_from_df(df)

    X_train, y_train, X_test, y_test, label_encoder = utils.split_df(df)
    X_train_no_rare, y_train_no_rare, X_test_no_rare, y_test_no_rare, _ = utils.split_df(df_no_rare)
    X_train_only_rare, y_train_only_rare, X_test_only_rare, y_test_only_rare, _ = utils.split_df(df_only_rare)

    client_names = ['{}_{}'.format("client", i+1) for i in range(constants.num_clients)]

    input_shape = X_train.shape[1:]
    nb_classes = len(label_encoder.classes_)
    class_weights = utils.get_class_weights(y_train)

    y_test = utils.convert_to_categorical(y_test, nb_classes)
    X_train_no_rare = utils.convert_to_categorical(X_train_no_rare, nb_classes)
    y_test_only_rare = utils.convert_to_categorical(y_test_only_rare, nb_classes)

    # this is used to test the global model with the whole test set
    test_batched_overall = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    
    # this is used to test the local model on a subset of the overall test set [UNION]
    test_batched_truflass = custom_extension.split_x_y_into_chunks(X_test, y_test, constants.comms_round)
    test_batched_truflass_only_rare = custom_extension.split_x_y_into_chunks(X_test_only_rare, y_test_only_rare, constants.comms_round)

    for r in runs:
        _percentage_small_clients = r["percentage_no_rares_clients"]
        special_client_amount = int(len(client_names) * _percentage_small_clients)
        random.shuffle(client_names)
        special_clients = client_names[:special_client_amount]
        print("--- --- --- STARTING NEW RUN --- --- ---")
        print("percentage_no_rares_clients: ", _percentage_small_clients)
        print("experiment_name: ", experiment_name)
        print("how_small_percentage: ", how_small_percentage)
        print(" --- --- --- ---- --- -- --- ---- ------ ---")
        run_single_case(experiment_name = experiment_name,
                        
                        client_names=client_names,

                        X_train = X_train, 
                        y_train = y_train,
                        X_test = X_test, 
                        y_test = y_test,

                        X_train_no_rare = X_train_no_rare, 
                        y_train_no_rare = y_train_no_rare,
                        X_test_no_rare = X_test_no_rare, 
                        y_test_no_rare = y_test_no_rare,

                        percentage_no_rares_clients = _percentage_small_clients,
                        how_small_percentage = how_small_percentage,
                        special_clients = special_clients, 
                        input_shape=input_shape, 
                        nb_classes=nb_classes, 
                        class_weights=class_weights, 
                        test_batched_overall=test_batched_overall, 
                        test_batched_truflass=test_batched_truflass,
                        test_batched_truflass_only_rare=test_batched_truflass_only_rare)
        
    play('./assets/alarm.mp3')
        