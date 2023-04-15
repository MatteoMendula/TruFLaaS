# TruFLaaS
## A novel Trustworthy Federated Learning Framework

TruFLaaS is a blockchain-based architecture that achieves Trustworthy federated learning as a service (TruFLaaS). Our solution provides trustworthiness among 3rd-party organizations by leveraging blockchain, smart contracts, and a decentralized oracle network. 

The table below highlights the main differences with the other state of the art solutions.




> This repository containes the code and the results obtained by running a testbed use case 
> written in Python, which simulates different node selection strategies under a broad 
> set of circumstances. 

Tu run each experiment simply install the required dependencies listed in requirements.txt and run the corresponding python script file.

## Experiments

- Heterogeneous Data Distribution: First, we consider a scenario where the distribution of data, among honest clients, is heterogeneous.
- Heterogeneous Data Distribution on Rare Cases: This experiment is a special case of the previous set. We focus on a deployment environment where some nodes have no data on a particular class of events, which we define as rare cases.
- Model Forging Attack: Then, we aim at evaluating the capability of TruFLaaS to be resilient against model forging attacks.

## Experiments (review extension)

The corresponding files can be found inside the "review" branch.

To address the review provided by the editorial board we extended our experiments by developing a novel multiclass classification for the N-BaIoT dataset. The network classifies the traffic data into 11 different categories.



TruFLaaS has been compared with both no node filtering techniques and well known state-of-the-art-solutions.
In particular, we have explored and analyzed different advantages offered by our solution if compared with weight up with [TrustFed](https://ieeexplore.ieee.org/document/9416805).


The results obtained show how our blockchain based approch led to better accuracies with a lower amount of aggregation rounds.
In addition, the identification of maliciuous or misliding nodes through rare cases testing proved to be a viable solution for making any Federated Learning more robust and resilient.



For any suggestion or contribution, or if you want to collaborate on some novel and intertaining project, don't hesitate to contact me at: matteo.mendula@unibo.it