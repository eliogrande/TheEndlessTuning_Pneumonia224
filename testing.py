import torch
from torch import nn
from torchvision import models
from tqdm import tqdm
from models_ import accuracy_fn
from dataloader import load_data

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device):
    test_loss, test_acc = 0, 0
    test_preds = []

    model.eval()
    with torch.no_grad():
        for batch_id, (X, y,_) in tqdm(enumerate(data_loader)):
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
            test_preds.append(y_pred.cpu().detach().numpy())

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    return test_loss, test_acc


device = torch.device('cpu')
print(f'Using {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
print(torch.__version__, torch.version.cuda+'\n')


test_dataloader,classes=load_data(batch_size=32,dataset_path='Case Study')
print('Number of classes: ',classes)


model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 14),
    nn.Linear(14,classes))
model.load_state_dict(torch.load('0316resnet18.pt')) 
model.to(device)
#print(model)


epochs = 1
loss_fn = nn.CrossEntropyLoss()

for e in range(epochs):

    current_test_loss, current_test_acc= test_step(
        data_loader = test_dataloader,
        model = model,
        loss_fn = loss_fn,
        accuracy_fn = accuracy_fn,
        device=device)

    # Logging
    print(('Epoch ' + str(e).rjust(2) + ':  ' +
        f'Test loss = {current_test_loss:.5f},  ' +
        f'Test acc = {current_test_acc:.2f}%'))

