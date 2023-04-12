import torch
import torch.optim as optim
import torch.nn as nn
import copy

from sklearn.metrics import f1_score


class WorkerFinalValidator():
    def __init__(self, ids, validation_DataLoader_overall, validation_Dataloader_rares):
        self.id = ids
        self.test_input_overall, self.test_output_overall = validation_DataLoader_overall
        self.test_input_rares, self.test_output_rares = validation_Dataloader_rares

    def test_other_model(self, model_1):
        criterion = nn.CrossEntropyLoss()
        model_1.eval()
        with torch.no_grad():
            output_overall = model_1(self.test_input_overall)
            output_overall = torch.squeeze(output_overall)
            loss_overall = criterion(output_overall, self.test_output_overall).item()

            output_rares = model_1(self.test_input_rares)
            output_rares = torch.squeeze(output_rares)
            loss_rares = criterion(output_rares, self.test_output_rares).item()

            preds_softmax_overall = torch.nn.functional.softmax(loss_overall, dim=1)
            pred_labels_overall = torch.argmax(preds_softmax_overall, dim=1)
            correct_preds_overall = torch.sum(pred_labels_overall == self.test_output)

            preds_softmax_rares = torch.nn.functional.softmax(loss_rares, dim=1)
            pred_labels_rares = torch.argmax(preds_softmax_rares, dim=1)
            correct_preds_rares = torch.sum(pred_labels_rares == self.test_output)

            print("correct_preds_overall", correct_preds_overall)
            print("len(self.test_output)", len(self.test_output_overall))

            # Compute accuracy
            f1_overall = f1_score(self.test_output_overall.numpy(), pred_labels_overall.numpy(), average='weighted')
            accuracy_overall = float(correct_preds_overall) / float(len(self.test_output_overall))


            f1_rares = f1_score(self.test_output_rares.numpy(), pred_labels_rares.numpy(), average='weighted')
            accuracy_rares = float(correct_preds_rares) / float(len(self.test_output_rares))


            # Print the shape of the resulting tensor

            # accuracy = accuracy_score(np.argmax(self.test_output, axis=0), np.argmax(output, axis=0))
            return {
                'loss': (loss_overall + loss_rares) / 2,
                'accuracy': (accuracy_overall + accuracy_rares) / 2,
                'f1': (f1_overall + f1_rares) / 2,
            }

    # def test_other_model(self, ids, model_1, results):
    #     criterion = nn.CrossEntropyloss_overall()
    #     model_1.eval()
    #     with torch.no_grad():
    #         output = model_1(self.test_input)
    #         output = torch.squeeze(output)
    #         loss_overall = criterion(output, self.test_output).item()
    #         results[self.id][ids] = loss_overall
    #         return loss_overall