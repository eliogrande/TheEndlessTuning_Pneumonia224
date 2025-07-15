import torch
from torch import nn
import numpy as np    
import joblib
from collections import defaultdict
from dataloader import load_data
from torchvision import models,transforms
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from PIL import Image

device = torch.device('cpu')
_,__,___,data= load_data(batch_size=1, dataset_path='Pneumonia_Or_Not') 

# FEATURE EMBEDDING *reduces images' dimension to 2
def get_embeddings():

    compressor = models.resnet18(pretrained=None)
    compressor.fc = nn.Sequential(
        nn.Linear(compressor.fc.in_features, out_features=14),
        nn.Linear(14,2))
    compressor.load_state_dict(torch.load('0316resnet18.pt', map_location=torch.device('cpu')))
    compressor = torch.nn.Sequential(*list(compressor.children())[:-1],
                                     nn.Flatten(),
                                     compressor.fc[0])
    
    compressor.to(device)
    pca = PCA(n_components=2)

    result = np.empty((0,14))
    iter_n = 0
    image_paths = []
    labels = []

    with torch.no_grad():
        for batch_id, (X, y, path) in enumerate(data):
            iter_n +=1
            print(f'Iteration: {iter_n}')
            X = X.to(device)
            y = y.to(device)
            labels.append(str(y[0].item()))
            y_pred = compressor(X)
            y_pred = y_pred.flatten(start_dim=1)
            y_pred = y_pred.numpy()       
            result = np.concatenate((result,y_pred),axis =0)
            path_normalized = path[0].replace("\\", "/")
            image_paths.append(path_normalized)
            #if iter_n == 3:
            #    break

    pca.fit(result)
    joblib.dump(pca, 'pca_model_RS18.pkl')
    X_reduced = pca.transform(result)
    zipped_data = list(zip(X_reduced, image_paths, labels))
    joblib.dump(zipped_data, 'embeddings_RS18.pkl')
    return None


def similarity_checker(path,num_points):
        
    case_study = Image.open(path).convert('RGB')
    transform = transforms.ToTensor()
    case_study = transform(case_study).unsqueeze(0)
    loaded_pca = joblib.load('pca_model_RS18.pkl')

    compressor = models.resnet18(pretrained=None)
    compressor.fc = nn.Sequential(
        nn.Linear(compressor.fc.in_features, out_features=14),
        nn.Linear(14,2))
    compressor.load_state_dict(torch.load('0316resnet18.pt', map_location=torch.device('cpu')))
    compressor = torch.nn.Sequential(*list(compressor.children())[:-1],
                                     nn.Flatten(),
                                     compressor.fc[0])
    compressor.to(device)

    with torch.no_grad():

        case_study.to(device)
        y_pred = compressor(case_study)
        y_pred = y_pred.flatten(start_dim=1)
        y_pred = y_pred.numpy()           
    
    CS_embedded = loaded_pca.transform(y_pred)

    embeddings = joblib.load('embeddings_RS18.pkl')

    #distanze dal case study
    distances = [euclidean(i[0],CS_embedded[0]) for i in embeddings]
    smallest_indices = [index for index, value in sorted(enumerate(distances), key=lambda x: x[1])[:num_points]] #10 istanze meno distanti dal case study
    zipped_distances = list(zip(embeddings,distances))
    
    x_values = []
    y_values = []
    cs_x = CS_embedded[0][0]
    cs_y = CS_embedded[0][1]
    img_address = []
    classes_ = []

    for i in smallest_indices:
        x_values.append(zipped_distances[i][0][0][0])
        y_values.append(zipped_distances[i][0][0][1])
        img_address.append(zipped_distances[i][0][1])
        classes_.append(zipped_distances[i][0][2])


    return x_values,y_values,cs_x,cs_y,img_address,classes_


if __name__ == 'main':

    get_embeddings()
    similarity_checker('./Pneumonia_Or_Not/0/1.jpg',num_points=3)