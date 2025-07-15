from timeit import default_timer as timer
import torch
from torch import nn
from collections import defaultdict
from dataloader import load_data,create_dataset
from models_ import Saver,accuracy_fn,top_x_accuracy,train_step,valid_step
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'Using {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
print(torch.__version__, torch.version.cuda+'\n')

# List all available model names
#model_names = dir(models)
#print(model_names)

 
# DATASET
#create_dataset(num_samples=2500,data_path='./Data_Source',resize=(224,224))
train_dataloader,valid_dataloader,classes,_= load_data(batch_size=32,dataset_path='Pneumonia_Or_Not') 
print('Number of classes: ',classes)


# HYPERPARAMETERS
lr = 5e-4
#momentum = 0.8
gamma = 0.7
max_epochs = 5
stop_epochs = 5
early_stop_patience = 3


# CLASSIFIER
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 14),
    nn.Linear(14, classes))
#model.load_state_dict(torch.load('0305resnet18.pt')) 
model.to(device)
print(model)

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, momentum=momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
saver = Saver('0316resnet18.pt') 
 
start = timer()
#results = defaultdict(list)
for e in range(max_epochs):

    # One epoch training of classifier
    print(f'current lr: {scheduler.get_lr()[0]}')
    current_train_loss, current_train_acc, train_top_x_acc = train_step(
        data_loader = train_dataloader,
        model = model,
        loss_fn = loss_fn,
        optimizer = optimizer,
        accuracy_fn = accuracy_fn,
        top_x_accuracy=top_x_accuracy,
        device=device)

    if e < stop_epochs:
        if scheduler: scheduler.step()

    # One epoch validation of classifier
    current_valid_loss, current_valid_acc, valid_top_x_acc = valid_step(
        data_loader = valid_dataloader,
        model = model,
        loss_fn = loss_fn,
        accuracy_fn = accuracy_fn,
        top_x_accuracy=top_x_accuracy,
        device=device)

    # Logging
    print(('Epoch ' + str(e).rjust(2) + ':  ' +
        f'Training loss = {current_train_loss:.5f},  ' +
        f'Training acc = {current_train_acc:.2f}%,  ' +
        f'Validation loss = {current_valid_loss:.5f},  ' +
        f'Validation acc = {current_valid_acc:.2f}%'))

    # Check if need to stop
    if saver.update_best_model(model, current_valid_loss, early_stop_patience):
        break

stop = timer()
saver.save()
print(f'Classifier training done in {stop-start} seconds')