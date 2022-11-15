import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError

class Worker():
    def __init__(self, ids, learning_rate, model, train_data, test_data):
        self.id = ids
        self.learning_rate = learning_rate
        self.model = model
        self.train_input, self.train_output = train_data
        self.test_input, self.test_output = test_data
        self.opt = optim.SGD(params=self.model.parameters(),lr=self.learning_rate)

    def train_my_model(self):
        criterion = nn.L1Loss()
        self.model.train()
        self.opt.zero_grad()
        pred = self.model(self.train_input)
        loss = criterion(pred, self.train_output)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def test_my_model(self):
        criterion = nn.L1Loss()
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.test_input)
            loss = criterion(output, self.test_output).item()
            return loss

    def test_other_model(self, ids, model_1, results):
        criterion = nn.L1Loss()
        model_1.eval()
        with torch.no_grad():
            output = model_1(self.test_input)
            loss = criterion(output, self.test_output).item()
            results[self.id][ids] = round(loss,3)
            return loss

    def test_final_model(self, model_1):
        mean_abs_percentage_error = MeanAbsolutePercentageError()
        criterion = nn.L1Loss()
        model_1.eval()
        with torch.no_grad():
            output = model_1(self.test_input)
            mae = criterion(output, self.test_output).item()
            mape = mean_abs_percentage_error(output, self.test_output).item()
            return (mae, mape)

    def set_weights(self, aggregated_numpy):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(aggregated_numpy[i]).type('torch.FloatTensor')