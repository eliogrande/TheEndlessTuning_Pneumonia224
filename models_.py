from timeit import default_timer as timer
from tqdm import tqdm
from collections import defaultdict
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms,models


class TrainingStoppedException(Exception):
    pass

class Saver:
    def __init__(self, path):
        self.path = path
        self.best_model = None
        self.min_validation_loss = float('inf')
        self.counter = 0

    def update_best_model(self, model, validation_loss, early_stop_patience):

        if validation_loss < self.min_validation_loss:
            print('Hey, new best model in town!')
            self.min_validation_loss = validation_loss
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'{self.counter} epoch with no loss decrease')
            if self.counter > early_stop_patience:
                print('Early stopped')
                return True 
        return False


    def save(self):
        print('Saving the best model')
        torch.save(self.best_model.state_dict(), self.path)

def accuracy_fn(y_true, y_pred):
    return (100 * torch.eq(y_true, y_pred).sum().item() / len(y_pred))

def top_x_accuracy(output,target,x=2):
    _, topx_preds = output.topk(x, dim=1, largest=True, sorted=True)
    correct = topx_preds.eq(target.view(-1, 1))
    correct_count = correct.sum().item()
    total_count = target.size(0)
    return correct_count / total_count * 100


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               top_x_accuracy,
               device):
    train_loss, train_acc,top_x_acc = 0, 0, 0

    model.train()
    for batch_id, (X, y,_) in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        # forward pass
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)

        # backward pass
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
        top_x_acc += top_x_accuracy(y_pred.detach(),y)

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    top_x_acc /= len(data_loader)

    return train_loss, train_acc, top_x_acc

def valid_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              top_x_accuracy,
              device):
    valid_loss, valid_acc, top_x_acc = 0, 0, 0
    valid_preds = []

    model.eval()
    with torch.no_grad():
        for batch_id, (X, y,_) in tqdm(enumerate(data_loader)):
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            valid_loss += loss_fn(y_pred, y).item()
            valid_acc += accuracy_fn(y, y_pred.argmax(dim=1))
            top_x_acc += top_x_accuracy(y_pred.detach(),y)
            valid_preds.append(y_pred.cpu().detach().numpy())

        valid_loss /= len(data_loader)
        valid_acc /= len(data_loader)
        top_x_acc /= len(data_loader)

    return valid_loss, valid_acc, top_x_acc



#---------------------------------------------------------------------
"""Sezione Autoencoder"""

def emb_train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    train_loss= 0

    model.train()
    for batch_id, (X, y) in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()

        # forward pass)
        X = X.to(device)
        y = model(X)

        # backward pass
        loss = loss_fn(y, X)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(data_loader)

    return train_loss

def emb_valid_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              device):
    valid_loss= 0

    model.eval()
    with torch.no_grad():
        for batch_id, (X, y) in tqdm(enumerate(data_loader)):
            X = X.to(device)
            y = model(X)
            valid_loss += loss_fn(y, X).item()

        valid_loss /= len(data_loader)

    return valid_loss