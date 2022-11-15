import torch
import time

import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy.random import default_rng
np.random.seed(1)

from worker import Worker
from worker import Validator
from net import Net
from utils import aggregate_model, process_data_final, select_node_to_discard_truflass
import random
import seaborn as sns
import time

from playsound import playsound as play
import pandas as pd

torch.manual_seed(1)



experiments = 1
iteration = 100
size = 20
num_workers = 100
n_validators = 100
rounds = 10

counter = 0
mae_performance_metrics = np.zeros(iteration)
mae_performance_metrics_truflass = np.zeros(iteration)

#questa funzione serve per rendere il plot più "smooth"
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

final_data = process_data_final()

#ho cambiato i batch size
train_standard = [(data, target) for _,(data, target) in enumerate(final_data["train_loader"])]
test_standard = [(data, target) for _,(data, target) in enumerate(final_data["test_loader"])]
#non è più tanto small, ho cambiato il batch_size
test_standard_small = [(data, target) for _,(data, target) in enumerate(final_data["test_loader_small"])]

def experiment_rare_cases(experiment_counter, n_specials):
    

    all_trainers = range(num_workers)
    malicious_nodes = random.sample(all_trainers, n_specials)
    benign_nodes = list(set(all_trainers) - set(malicious_nodes))

    print("-----------------------------------")
    print("malicious_nodes", malicious_nodes)
    print("-----------------------------------")
    print("benign_nodes", benign_nodes)


    model_original = Net()
    # nodi per esperimento con truflaas
    workers_malicious = {}
    workers_benign = {}
    # nodi esperimento senza truflaas
    workers = {}
    





    # workers per esperimento sneza protocollo truflass => i nodi malevoli vengono inizializzati insieme agli altri
    for w in all_trainers:

        data_train = train_standard[w][0]
        target_train = train_standard[w][1]

        workers[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (None, None))


    # nodi malevoli sia con truflaas che senza
    for w in malicious_nodes:

        #creo dati di training con rumore (come fa TrustFed). 15 è quello che loro chiamano fattore di rumore
        data_train = torch.rand(data_train.shape[0], data_train.shape[1], data_train.shape[2]) * 15
        target_train = train_standard[w][1]

        #truflaas
        workers_malicious[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (None, None))

        #no truflaas
        workers[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (None, None))

    
    #ndoi benigni per truflaas
    for w in benign_nodes:
        data_train = train_standard[w][0]
        target_train = train_standard[w][1]

        workers_benign[w] = Worker(w, 0.1, copy.deepcopy(model_original), (data_train, target_train), (None, None))
        

    # nodo validatore, i batch in test_standard_small li usa per testare ad ogni round i modelli, 
    # test_standard[0] è un unico batch per vedere le perfomance del modello globale (non diventa pubblico)
    tester =  Validator(test_standard_small, test_standard[0])


    csv_results = [["Iteration", "no_special_mae", "no_special_mape", "no_filter_mae", "no_filter_mape", "trustfed_mae", "trustfed_mape", "truflass_mae", "truflass_mape"]]

    # ------------------------------------------- learning starts here
    time_experiment = int(time.time())

    # tengo traccia delle transazioni accettate
    tx_accepted = np.zeros(num_workers)
    for itr in range(iteration):

        print(f'--------------------- Iteration {itr} ---------------------')
        for node in malicious_nodes:
            for _ in range(rounds):
                loss = workers_malicious[node].train_my_model()

        for node in benign_nodes:
            for _ in range(rounds):
                loss = workers_benign[node].train_my_model()
        for node in all_trainers:
            for _ in range(rounds):
                loss = workers[node].train_my_model()



        
        results_truflass = []
        results_no_truflass = []
        for w in malicious_nodes:
            # al tester passo l'iterazione attuale solo per prendere un batch dal validation set
            loss_truflass = tester.test_model(copy.deepcopy(workers_malicious[w].model), itr)
            results_truflass.append((w,loss_truflass))
        for w in benign_nodes:
            loss_truflass = tester.test_model(copy.deepcopy(workers_benign[w].model), itr)
            results_truflass.append((w,loss_truflass))
        for w in all_trainers:
            results_no_truflass.append(tester.test_model(copy.deepcopy(workers[w].model), itr))




        malicious_detected_truflass = select_node_to_discard_truflass(results_truflass)
            

        print("malicious_detected_truflass", malicious_detected_truflass)


        truflass_models = []
        models = []
        for w in malicious_nodes:
            if not w in malicious_detected_truflass:
                tx_accepted[w] += 1 
                # aggiungo modello e peso del modello
                truflass_models.append((workers_malicious[w].model, tx_accepted[w]/(itr+1)))
                
        for w in benign_nodes:
            if not w in malicious_detected_truflass:
                tx_accepted[w] += 1 
                truflass_models.append((workers_benign[w].model, tx_accepted[w]/(itr+1)))
        for w in all_trainers:
            # qua i pesi sono tutti 1
            models.append((workers[w].model,1))


        general_model_truflass = aggregate_model(truflass_models)
        general_model = aggregate_model(models)

        gm_truflass = Net()
        for i, param in enumerate(gm_truflass.parameters()):
            param.data = torch.from_numpy(general_model_truflass[i]).type('torch.FloatTensor')
        

        gm = Net()
        for i, param in enumerate(gm.parameters()):
            param.data = torch.from_numpy(general_model[i]).type('torch.FloatTensor')


        # set aggregated weights on workers
        for w in malicious_nodes:
            workers_malicious[w].set_weights(general_model_truflass)
        for w in benign_nodes:
            workers_benign[w].set_weights(general_model_truflass)

        for w in all_trainers:
            workers[w].set_weights(general_model)
        

        performance_truflass = tester.test_final_model(gm_truflass)
        performance = tester.test_final_model(gm)

        # prendo MAE
        mae_performance_metrics_truflass[itr] = performance_truflass[0]
        mae_performance_metrics[itr] = performance[0]

        this_run_results = []
        this_run_results.append(itr)
        this_run_results.append(mae_performance_metrics)
        csv_results.append(this_run_results)

    plt.figure(figsize = (5,5))
    plt.xlabel('Rounds')
    plt.ylabel('MAE')
    # rendo più smooth i valori
    smoothed_mae = np.array(smooth(mae_performance_metrics,0.9))
    smoothed_mae_truflaas = np.array(smooth(mae_performance_metrics_truflass,0.9))
    plt.plot(range(iteration), smoothed_mae, label='MAE')
    plt.plot(range(iteration), smoothed_mae_truflaas, label='MAE TruFLaaS')

    plt.legend()

    plt.savefig(f'./results/{n_specials}/{time_experiment}.png', format="png", bbox_inches='tight')

    pd.DataFrame(mae_performance_metrics).to_csv(f'./results/{n_specials}/mae_performance_metrics_{time_experiment}.csv')
    pd.DataFrame(mae_performance_metrics_truflass).to_csv(f'./results/{n_specials}/mae_performance_metrics_truflaas_{time_experiment}.csv')
    pd.DataFrame(smoothed_mae).to_csv(f'./results/{n_specials}/smoothed_mae_{time_experiment}.csv')
    pd.DataFrame(smoothed_mae_truflaas).to_csv(f'./results/{n_specials}/smoothed_mae_truflaas_{time_experiment}.csv')


    
if __name__ == '__main__':
    
    for n_specials in [10, 25, 40]:
        for experiment_counter in range(experiments):
            print("-----------------------------------")
            print("running experiment:", experiment_counter)
            print("-----------------------------------")
            experiment_rare_cases(experiment_counter, n_specials)
                    
        counter += 1

    
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