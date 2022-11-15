import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError
import copy


class WorkerValidator():
    def __init__(self, ids, validation_DataLoader):
        self.id = ids
        self.validation_DataLoader = validation_DataLoader
        self.validation_counter = 0

    def test_other_model(self, workers_to_check, results):
        validation_input = self.validation_DataLoader[self.validation_counter][0]
        validation_output = self.validation_DataLoader[self.validation_counter][1]
        criterion = nn.L1Loss()
        losses = []
        for w in workers_to_check:
            id_w = workers_to_check[w].id
            model_w = copy.deepcopy(workers_to_check[w].model)
            model_w.eval()
            with torch.no_grad():
                output = model_w(validation_input)
                loss = criterion(output, validation_output).item()
                losses.append(loss)
                results[id_w] = round(loss,3)
        # print("validor node, validation", self.validation_counter)
        self.validation_counter += 1
        if self.validation_counter >= len(self.validation_DataLoader):
            self.validation_counter = 0
        return loss
