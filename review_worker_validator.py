import torch
import torch.optim as optim
import torch.nn as nn
import copy


class WorkerValidator():
    def __init__(self, ids, validation_DataLoader):
        self.id = ids
        self.validation_DataLoader = validation_DataLoader
        self.validation_counter = 0

    def test_other_model(self, workers_to_check, results):
        validation_x = self.validation_DataLoader[self.validation_counter][0]
        validation_y = self.validation_DataLoader[self.validation_counter][1]
        criterion = nn.CrossEntropyLoss()
        losses = []
        for w in workers_to_check:
            id_w = workers_to_check[w].id
            model_w = copy.deepcopy(workers_to_check[w].model)
            model_w.eval()
            with torch.no_grad():
                output = model_w(validation_x)
                output = torch.squeeze(output)
                # print("output.shape: ", output.shape)
                # print("validation_y.shape: ", validation_y.shape)
                # print("output: ", output)
                # print("validation_y: ", validation_y)
                loss = criterion(output, validation_y).item()
                losses.append(loss)
                results[id_w] = loss
        self.validation_counter += 1
        if self.validation_counter >= len(self.validation_DataLoader):
            self.validation_counter = 0
        return loss
    

    # def test_other_model(self, ids, model_1, results):
    #     criterion = nn.CrossEntropyLoss()
    #     model_1.eval()
    #     with torch.no_grad():
    #         output = model_1(self.test_input)
    #         output = torch.squeeze(output)
    #         loss = criterion(output, self.test_output).item()
    #         results[self.id][ids] = loss
    #         return loss