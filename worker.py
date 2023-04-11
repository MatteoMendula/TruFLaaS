import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

import numpy as np

class Worker():
    def __init__(self, ids, learning_rate, model, train_data, test_data):
        self.id = ids
        self.learning_rate = learning_rate
        self.model = model
        self.train_input, self.train_output = train_data
        self.test_input, self.test_output = test_data
        self.opt = optim.SGD(params=self.model.parameters(),lr=self.learning_rate)

    def train_my_model(self):
        # criterion = nn.L1Loss()
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        self.opt.zero_grad()
        pred = self.model(self.train_input)
        pred = torch.squeeze(pred)
        # print("pred shae", pred.shape)
        # print("pred", pred)
        # print("train output shape", self.train_output.shape)
        # print("train output", self.train_output)
        loss = criterion(pred, self.train_output)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def test_my_model(self):
        criterion = nn.L1Loss()
        # criterion = nn.CrossEntropyLoss()
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.test_input)
            output = torch.squeeze(output)
            loss = criterion(output, self.test_output).item()
            return loss

    def test_other_model(self, ids, model_1, results):
        # criterion = nn.L1Loss()
        criterion = nn.CrossEntropyLoss()
        model_1.eval()
        with torch.no_grad():
            output = model_1(self.test_input)
            output = torch.squeeze(output)
            # print("output shape", output.shape)
            # print("self test output shape", self.test_output.shape)
            loss = criterion(output, self.test_output).item()
            results[self.id][ids] = loss
            return loss

    def test_final_model(self, model_1):
        # criterion = nn.L1Loss()
        criterion = nn.CrossEntropyLoss()
        model_1.eval()
        with torch.no_grad():
            output = model_1(self.test_input)
            output = torch.squeeze(output)
            loss = criterion(output, self.test_output).item()
            
            print("output", output)
            print("shape", output.shape)
            print("self test output", self.test_output)
            print("shape", self.test_output.shape)

            softmax_tensor = torch.nn.functional.softmax(output, dim=1)

            # Print the shape of the resulting tensor
            print(softmax_tensor.shape)

            # accuracy = accuracy_score(np.argmax(self.test_output, axis=0), np.argmax(output, axis=0))
            return (None, loss)

    def set_weights(self, aggregated_numpy):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.from_numpy(aggregated_numpy[i]).type('torch.FloatTensor')