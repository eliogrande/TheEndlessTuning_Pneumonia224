import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms,models
from torch.utils.data import DataLoader
from models_ import Saver,accuracy_fn,train_step,valid_step,top_x_accuracy
from timeit import default_timer as timer
from tqdm import tqdm
from collections import defaultdict
from dataloader import ImageFolderWithPaths


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'Using {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print(torch.__version__, torch.version.cuda+'\n')

# Definire le trasformazioni (se necessarie, come ridimensionamento, normalizzazione ecc.)
transform = transforms.ToTensor()

# Caricare il dataset
dataset = ImageFolderWithPaths(root='./tuning', transform=transform)

# Dividere il dataset in training e test (opzionale)
train_size = int(0.7 * len(dataset))  # 80% dei dati per l'addestramento
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Creare i DataLoader
train_dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=3, shuffle=False)

# Verifica
for images, labels,_ in train_dataloader:
    print(images.shape)  # Output: batch_size x canali x altezza x larghezza
    print(labels.shape)
    print(images[0].min(),images[0].max())
    break

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 14),
    nn.Linear(14, 2))
model.load_state_dict(torch.load('0316resnet18.pt',map_location=device))

# HYPERPARAMETERS
lr = 5e-6
gamma = 1
max_epochs = 1
stop_epochs = 0
early_stop_patience = 3

# CLASSIFIER
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
saver = Saver('0318resnet18_updated.pt')

start = timer()
results = defaultdict(list)
for e in range(max_epochs):
    # One epoch training of classifier
    print(f'current lr: {scheduler.get_last_lr()[0]}')
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
        f'Class_training loss = {current_train_loss:.3f},  ' +
        f'Class_training acc = {current_train_acc:.2f}%,  ' +
        f'Class_validation loss = {current_valid_loss:.3f},  ' +
        f'Class_validation acc = {current_valid_acc:.2f}%'))

    # Check if need to stop
    if saver.update_best_model(model, current_valid_loss, early_stop_patience):
        break

stop = timer()
saver.save()
print(f'Classifier tuning done in {stop-start} seconds')