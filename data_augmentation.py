from PIL import Image, ImageEnhance
import random
import os
import shutil

"""Selezionare altre 1000 immagini da mnist (PROVVISORIAMENTE), che il modello non ha visto. A questo punto
prendere un case study (eventualmente flipparlo di modo da ricavarne altri due o 3, ma qui non è implementato 
per problemi di bilanciamento di classe), poi prendere randomicamente per ogni classe, da quelle nuove 1000 immagini, 
un'immagine per ogni altra classe che non sia quella del case study.

NOTA: il codice è meramente sperimentale e non è fatto per tornare sulla stessa immagine prima del fine-tuning. Se
l'utente ricarica due volte il medesimo case study e dà due etichette diverse, saranno caricateo 20 immagini di cui
le due interessate con due etichette diverse, ingenerando confusione nel modello. Bisognerebbe implementare un registro
dei file caricati, eliminare il case study precedente dalla cartella tuning insieme alla sua selezione randomica
e rifare la selezione randomica col case_study ripreso, con un'etichetta diversa.

La decisione dell'utente, dunque, nei termini dell'esperimento, dev'essere DEFINITIVA. Per un'implementazione più seria
dovrà essere resa provvisoria."""


def augment_case_study(img_path,temporary_folder,class_to_exclude):
    basename, extension = os.path.splitext(img_path)
    #print('img_path: ',img_path)
    #print(extension)
    
    for subfolder in os.listdir(temporary_folder):
            if subfolder != class_to_exclude:
                if not os.path.exists('./tuning/'+subfolder):
                    os.makedirs('./tuning/'+subfolder)
                filenames = os.path.join(temporary_folder,subfolder)
                add_class_item = random.sample(os.listdir(filenames),1)[0]
                print('filenames: ',filenames)
                print('basename: ',basename.split('/')[-2])
                print('subfolder: ',subfolder)
                if basename.split('/')[-2] != subfolder:
                    print('Basename is different from subfolder')
                    with Image.open(os.path.join(filenames,add_class_item)) as x:
                        print('Adding item to tuning set...')
                        x.save('./tuning/'+os.path.join(subfolder,add_class_item))
                        os.remove(os.path.join(filenames,add_class_item))
                    
#augment_case_study(img_path="./tuning/1/100.jpg",temporary_folder='./temporary_images')