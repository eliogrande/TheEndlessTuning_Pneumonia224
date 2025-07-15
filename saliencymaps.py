import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms,models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from RISE.explanations import RISE
from RISE.utils import *


def load_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image


def explain_gradcam(image_path, model):
    
    input_image = load_image(image_path)
    grad_cam = GradCAM(model, 
                       target_layers=[model.layer4[-1]])  # 'layer4' è l'ultimo layer di convoluzione in ResNet-50
    activation_map = grad_cam(input_image)
    activation_map = activation_map.squeeze(0)
    original_image = Image.open(image_path)
    original_image = np.array(original_image)

    plt.figure()
    plt.gcf().set_facecolor('#101010')  # Sfondo figura
    plt.gca().set_facecolor('#101010')   #sfondo plot
    plt.imshow(original_image,alpha=1, aspect='auto')
    plt.imshow(activation_map,cmap='jet',alpha=0.7,aspect='auto')
    colorbar = plt.colorbar(orientation='vertical', shrink=1)
    colorbar.set_ticks(colorbar.get_ticks())  # Per applicare modifiche ai tick
    colorbar.ax.tick_params(axis='y', colors='white')  # Cambia il colore del testo dei tick in bianco
    plt.title('Saliency Map', fontsize=12, color='white')
    plt.tick_params(axis='both', colors='white')
    plt.savefig("./temp/case_study_gradcam.jpg", format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()


def explain_rise(image_path,model,N,fit_callback,exp_callback,device):
    
    explainer = RISE(model, input_size=(224,224), gpu_batch=10)
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = True
    img = read_tensor(image_path)
    img = img.to(device)
    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=N, s=8, p1=0.1, savepath=maskspath, fit_callback=fit_callback)
    else:
        explainer.load_masks(maskspath)
        print('Masks are loaded.')
    
    saliency = explainer(img, exp_callback).cpu().numpy()
    
    plt.figure()
    plt.gcf().set_facecolor('#101010')  # Sfondo figura
    plt.gca().set_facecolor('#101010')   #sfondo plot
    plt.imshow(img.squeeze(0).permute(1,2,0).cpu().numpy(),alpha=1, aspect='auto')
    plt.imshow(saliency[0],cmap='jet',alpha=0.5,aspect='auto')
    colorbar = plt.colorbar(orientation='vertical', shrink=1)
    colorbar.set_ticks(colorbar.get_ticks())  # Per applicare modifiche ai tick
    colorbar.ax.tick_params(axis='y', colors='white')  # Cambia il colore del testo dei tick in bianco
    plt.title('Saliency Map', fontsize=12, color='white')
    plt.tick_params(axis='both', colors='white')
    plt.savefig("./temp/case_study_rise.jpg", format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    #plt.show()

def fit_callback(x):
    pass
def explain_callback(x):
    pass


if __name__  == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, out_features=14),
        nn.Linear(14,2))
    if device == 'cuda':
        model.load_state_dict(torch.load('0316resnet18.pt'))#, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('0316resnet18.pt', map_location=torch.device('cpu')))
    model.to(device)
    torch.manual_seed(7)
    
    #explain_gradcam(image_path='./case_studies/2/127.jpg',model=model)
    #explain_rise(image_path='./case_studies/1/389.jpg',model=model,N=1500,device=device,fit_callback=fit_callback,exp_callback=explain_callback)
