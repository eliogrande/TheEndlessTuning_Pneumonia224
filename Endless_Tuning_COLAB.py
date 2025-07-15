import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import streamlit as st
import torch
from torch import nn
from torchvision import models
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from embedding import similarity_checker
from saliencymaps import load_image,explain_gradcam,explain_rise
from data_augmentation import augment_case_study
from PIL import Image  
import os
import csv
from datetime import datetime
import base64
from io import BytesIO
import shutil


@st.cache_resource#(allow_output_mutation=True)
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 14),
        nn.Linear(14, 2))
    if device == 'cuda':
        model.load_state_dict(torch.load('0316resnet18.pt'))#, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('0316resnet18.pt', map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    torch.manual_seed(7)
    return model,device

# Carica il modello una sola volta
model,device = load_model()
print("ID del modello:", id(model))


def go_to_page(pagina_dest):
    st.session_state.pagina = pagina_dest

def log_interaction(interaction_type, details=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('user_interactions.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, interaction_type, details])
    success = 'OK'
    return success

def image_to_base64(image_path):
    """Convert image file to base64 string."""
    buffered = BytesIO()
    image = Image.open(image_path)
    image.save(buffered, format="PNG")  #PNG
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_bytes(image_path, format='PNG'):
    with Image.open(image_path) as img:
        buffer = BytesIO()
        img.save(buffer, format=format)
        image_bytes = buffer.getvalue()
    return image_bytes




####    INTRO   ###############################
def pagina_1():
    if os.path.exists('./user_interactions.csv'):
        os.remove('./user_interactions.csv')

    st.markdown("<h1 style='text-align: center;'>The Endless Tuning</h1>", unsafe_allow_html=True) 

    # Box informativo
    st.markdown("""
        <div style="background-color: #000000; border-radius: 1vw; padding: 2vw; margin-top: 2vw;">
            <p style="font-size: 100%; text-align: justify;">This is the Endless Tuning, a relational approach to artificial intelligence based on
                a continuous interaction and a double-mirror feedback. Here, in particular, in a decision-making setting for
                image classification. Firstly, you will be asked to load an image. If you will feel confident about the case
                you will be simply asked of providing the right label. If otherwise you will feel unconfident,
                some suggestions and clues coming from the machine will nudge you to reflect. At the end of the process, you will have the last word.
                After a sufficient number of sessions, the model will automatically update its parameters, 
                so as to "tune" with your own style and learn together with you. You don't have any time constraint</p>
                <p style="font-size: 100%; text-align: justify;"> Note that <strong>every</strong> interaction will be recorded.
                However, records will be totaly anonymous, deleted at the end of every cycle. In the end, you will be able to download your data.
                By clicking on the "I understand" button, you declare that you understand and agree.</p>
        </div>
    """, unsafe_allow_html=True)
 
    cola,colb = st.columns([0.7,1.05])
    with colb:
        if st.button("I understand"):
            log_interaction("pressed key I understand")
            go_to_page(2) 
            st.rerun()


####    LOADING IMAGE   ##################################
def pagina_2():

    torch.cuda.empty_cache()

    if 'notes3' in st.session_state:
        del st.session_state.notes3
    if 'notes4' in st.session_state:
        del st.session_state.notes4
    if 'notes5' in st.session_state:
        del st.session_state.notes5
    if 'notes6' in st.session_state:
        del st.session_state.notes6
    if 'notes7' in st.session_state:
        del st.session_state.notes7


    st.title("setting #3: pneumonia")

    images_folder = '/content/TheEndlessTuning/ET_Pneumonia_224px/case_studies/'
    sottocartelle = [d for d in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, d))]
    selected_sottocartella = st.selectbox("Select a subfolder", sottocartelle)
    if selected_sottocartella:
        image_folder = os.path.join(images_folder, selected_sottocartella)
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        selected_image = st.selectbox("Select an image", image_files)
        uploaded_path = os.path.join(image_folder, selected_image)

        colx,coly,colz = st.columns([1,1,1])
        with coly:
    #uploaded_path = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg","bmp"])
          if uploaded_path is not None:
      #    if uploaded_path.type in ["image/png", "image/jpeg", "image/jpg", "image/bmp"]:
      #       print(uploaded_path,uploaded_path.name)
              img = Image.open(uploaded_path)
              st.image(img, use_container_width=True)
              st.session_state.uploaded_path = uploaded_path
              if 'log1' not in st.session_state:
                  st.session_state.log1 = log_interaction("Loaded image", uploaded_path)
              if os.path.exists('./temp'):
                  shutil.rmtree('./temp')    
              img_dir = './temp'
              if not os.path.exists(img_dir):
                  os.makedirs(img_dir)
              img.save(img_dir+'/case_study.jpg')
          else:
              st.write("An error occurred!")

    cola,colb = st.columns([6.6,1])
    with cola:
        if st.button("Back"):
            if 'log2' not in st.session_state:
                st.session_state.log2 = log_interaction("Going back...")
            go_to_page(1)
            st.rerun()  
    with colb:
        proceedp2 = st.button('Proceed')
    if proceedp2:
        if uploaded_path is None:
            log_interaction("Error: tried to proceed without loading file")
            st.markdown("<p style='font-size: 1vw;'>You firstly need to upload a file!</p>", unsafe_allow_html=True)
        else:
            if 'log3' not in st.session_state:
                st.session_state.log3 = log_interaction("Proceeding to next page...")
            go_to_page(3)  
            st.rerun()  


####    FIRST IMPRESSION    #################################
def pagina_3():
    if 'log1' in st.session_state:
        del st.session_state.log1
    if 'log2' in st.session_state:
        del st.session_state.log2
    if 'log3' in st.session_state:
        del st.session_state.log3

    st.title("What's your impression?")
    st.write("""Speak your mind! Take your time to observe the image, 
    then select the purported class. You may decide whether to go on 
    or stop here and go to the next session. If you feel confident, 
    press 'Save and exit'. Otherwise, if you think you could receive 
    useful suggestions from artificial intelligence, please proceed.""")
    st.markdown("<br>", unsafe_allow_html=True) 
    if "uploaded_path" in st.session_state:
        img = Image.open(st.session_state.uploaded_path)
        col_img, col_notes = st.columns([5,5])
        with col_img:
            st.image(img, use_container_width=True)
        with col_notes:
            if 'notes3' not in st.session_state:
                st.session_state.notes3 = "" 
            st.session_state.notes3 = st.text_area("Add notes. They'll be recorded.",
                                                  value=st.session_state.notes3, height=300)
    
    selected_class = st.selectbox("Select a class:", options)
    st.markdown("<br>", unsafe_allow_html=True) 

    cola,colb,colc = st.columns([4.5,5,1.5])
    with cola:
        if st.button("Back"):
            if 'log4' not in st.session_state:
                st.session_state.log4 = log_interaction("temporary_notes_3", st.session_state.notes3) 
            if 'log5' not in st.session_state:
                st.session_state.log5 = log_interaction(f'First impression: {selected_class}')
            log_interaction("Going back...")  
            del st.session_state.uploaded_path
            go_to_page(2)
            st.rerun()  
    with colb:
        savep3 = st.button('Save and Exit')
    if savep3:
    
        if selected_class != 'None':
    
            if 'log4' not in st.session_state:
                st.session_state.log4 = log_interaction("temporary_notes_3", st.session_state.notes3) 
            if 'log5' not in st.session_state:
                st.session_state.log5 = log_interaction(f'First impression: {selected_class}')

            dir_path = os.path.join(os.getcwd(),'tuning/',class_dict[selected_class])
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    #st.success(f"Directory './tuning/{selected_class}' created.")
                except Exception as e:
                    st.error(f"Error creating directory: {str(e)}")

            files = os.listdir(dir_path)
            jpg_files = [f for f in files if f.endswith('.jpg') and f.split('.')[0].isdigit()]

            if jpg_files:
                max_num = max(int(f.split('.')[0]) for f in jpg_files)
                next_file = f"{max_num + 1}.jpg"
            else:
                next_file = "0.jpg"
        
            with open(os.path.join(dir_path,next_file), "wb") as f:
                f.write(get_image_bytes(st.session_state.uploaded_path))
                
            augment_case_study(img_path=os.path.join(dir_path,next_file),
                                   temporary_folder='./temporary_images',class_to_exclude=class_dict[selected_class])

            go_to_page(10)
            log_interaction("Saved their impression at page 3", st.session_state.notes3)  
            st.rerun()
        else:
            st.markdown("<p style='font-size: 1vw;'>You firstly need to select a class!</p>", unsafe_allow_html=True)
    with colc:
        proceedp3 = st.button("Proceed")
    if proceedp3:
        if selected_class != 'None':

            if 'log4' not in st.session_state:
                st.session_state.log4 = log_interaction("temporary_notes_3", st.session_state.notes3) 
            if 'log5' not in st.session_state:
                st.session_state.log5 = log_interaction(f'First impression: {selected_class}')
            go_to_page(4)
    
            st.rerun()
        else:
            st.markdown("<p style='font-size: 1vw;'>You firstly need to select a class!</p>", unsafe_allow_html=True)

 
####    HINTS   ####################################
## RISE ##
def pagina_4():        
    if 'log4' in st.session_state:
        del st.session_state.log4
    if 'log5' in st.session_state: 
        del st.session_state.log5

    st.title("First clue")
    st.write("""Now, observe some clouds of possible significance on the surface
    of the image, extracted with the RISE algorithm. In order to get more precise
    suggestions, processing will take a while. Then take your time and update or 
    confirm your hypothesis. Look at the legend: the more intensely red the colour, 
    the higher the importance (excluding black padding).
    
    Warning: the model might have paid attention to irrelevant factors, 
    and the maps might not be entirely faithful. Take them just as 
    cause for reflection.
    """)

    col1, col2 = st.columns([0.6,0.3])
    with col1: 

        if 'rise' not in st.session_state:

            def update_progress_bar_fit(step,progress_bar=st.progress(0),progress_text=st.empty()):
                progress_value = int((step / rise_N) * 100)  
                progress_bar.progress(progress_value)  
                progress_text.text(f"Fitting: {progress_value}%")  
                if progress_value == 100:
                    progress_bar.empty()
                    progress_text.empty()

            def update_progress_bar_exp(step,progress_bar=st.progress(0),progress_text=st.empty()):
                progress_value = int((step / rise_N) * 100) 
                progress_bar.progress(progress_value)  
                progress_text.text(f"Elaborating suggestion: {progress_value}%")  
                if progress_value == 100:
                    progress_bar.empty()
                    progress_text.empty()

        
            print('Running RISE algorithm...')
            explain_rise(image_path='./temp/case_study.jpg',
                         fit_callback=update_progress_bar_fit,
                         exp_callback=update_progress_bar_exp,
                         N=rise_N,
                         model=model,
                         device=device) 
            st.session_state.rise = Image.open('./temp/case_study_rise.jpg') 
        st.image(st.session_state.rise, use_container_width=True)

    with col2:  
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.image(st.session_state.uploaded_path, use_container_width=True)

    st.write('Your previous annotations:')
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
    
    if 'notes4' not in st.session_state:
        st.session_state.notes4 = "" 
    st.session_state.notes4 = st.text_area("Add notes.They'll be recorded.",
                                            value=st.session_state.notes4, height=100)
    selected_class = st.selectbox("Select a class:", options)


    cola, colb = st.columns([10, 1.5])
    with cola:
        if st.button("Back"):
            if 'log6' not in st.session_state:
                st.session_state.log6 = log_interaction("temporary_notes_4", st.session_state.notes4) 
            if 'log7' not in st.session_state:
                st.session_state.log7 = log_interaction(f'Second impression: {selected_class}')
            log_interaction("Going back...")  
            del st.session_state.rise
            if 'gradcam' in st.session_state:
                del st.session_state.gradcam
            go_to_page(3)
            st.rerun()  
    with colb:
        proceedp4 = st.button('Proceed')
    if proceedp4:
        if selected_class != 'None':
            if 'log6' not in st.session_state:
                st.session_state.log6 = log_interaction("temporary_notes_4", st.session_state.notes4) 
            if 'log7' not in st.session_state:
                st.session_state.log7 = log_interaction(f'Second impression: {selected_class}')
            go_to_page(5)
            st.rerun()
        else:
            st.markdown("<p style='font-size: 1vw;'>You firstly need to select a class!</p>", unsafe_allow_html=True)


#### GRADCAM ###########################################################################
def pagina_5():        

    if 'log6' in st.session_state:
        del st.session_state.log6
    if 'log7' in st.session_state:
        del st.session_state.log7    

    st.title("Second clue")
    st.write("""Now, observe some clues of possible significance on the surface of the image, 
    extracted through the Grad-CAM algorithm. Look at the legend: the more intensely red the colour, 
    the higher the importance (excluding black padding). Take your time, then update or confirm 
    your hypothesis.
             
    Warning: the model might have paid attention to irrelevant factors, 
    and the maps might not be entirely faithful. Take them just as 
    cause for reflection.""")

    col1, col2 = st.columns([0.7,0.3])  
    with col1:   
        if 'gradcam' not in st.session_state:
            with st.spinner('Loading the saliency map...'):
    
                print('Running GRADCAM explainer...')
                explain_gradcam(image_path='./temp/case_study.jpg',model=model) 
                st.session_state.gradcam = Image.open('./temp/case_study_gradcam.jpg') 
        st.image(st.session_state.gradcam, use_container_width=True)

    with col2: 
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.image(st.session_state.uploaded_path, use_container_width=True)
    
    st.write('Your previous annotations:')
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes4}</h3>", unsafe_allow_html=True)

    if 'notes5' not in st.session_state:
        st.session_state.notes5 = "" 
    st.session_state.notes5 = st.text_area("Add your notes here. They'll be recorded.",
                                            value=st.session_state.notes5, height=100)
    selected_class = st.selectbox("Select a class:", options)
    
    cola, colb = st.columns([10, 1.5])
    with cola:
        if st.button("Back"):
            if 'log8' not in st.session_state:
                    st.session_state.log8 = log_interaction("temporary_notes_5", st.session_state.notes5) 
            if 'log9' not in st.session_state:
                    st.session_state.log9 = log_interaction(f'Third impression: {selected_class}')
            log_interaction("Going back...")  
            go_to_page(4)
            st.rerun()  
    with colb:
        proceedp5 = st.button('Proceed')
        if proceedp5:
            if selected_class != 'None':
                if 'log8' not in st.session_state:
                    st.session_state.log8 = log_interaction("temporary_notes_5", st.session_state.notes5) 
                if 'log9' not in st.session_state:
                    st.session_state.log9 = log_interaction(f'Third impression: {selected_class}')
                go_to_page(6)
                st.rerun()
            else:
                st.markdown("<p style='font-size: 1vw;'>You firstly need to select a class!</p>", unsafe_allow_html=True)



####    SIMILARITY  ############################################
def pagina_6():

    if 'log8' in st.session_state:
        del st.session_state.log8
    if 'log9' in st.session_state:
        del st.session_state.log9

    st.title("Similarity")
    st.write("""Here the 3 most similar cases and corresponding 
    labels taken from the original dataset, according to compressed 
    representations obtained through Principal Component Analysis 
    and plotted in their relative position. Hover on the points to
    get more precise information. Zoom + or - if necessary, 
    then update or confirm your hypothesis.""")

    x_values, y_values, cs_x, cs_y, images, classes = similarity_checker(
        path=st.session_state.uploaded_path,
        num_points=3
    )

    similar_imgs = []
    for i in range(len(x_values)):
        img_path = os.path.abspath(images[i])
        #print(img_path)
        similar_imgs.append(img_path)

    col3,col4,col5 = st.columns([1,1,1])
    with col3:
        st.write('Rank: #1')  
        similar_imgs_0 = Image.open(similar_imgs[0])
        st.image(similar_imgs_0, use_container_width=True)
    with col4:
        st.write('Rank: #2')
        similar_imgs_1 = Image.open(similar_imgs[1]) 
        st.image(similar_imgs_1, use_container_width=True)
    with col5:
        st.write('Rank: #3')
        similar_imgs_2 = Image.open(similar_imgs[2])
        st.image(similar_imgs_2, use_container_width=True)

    col2, col1 = st.columns([0.3, 0.6])
 
    with col1:
        with st.spinner('Loading...'):

            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[cs_x],  
                y=[cs_y],  
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='circle'),
                text=['Your image'],
                textposition='top right',  
                textfont=dict(color='red', size=14),
            ))
            for i in range(len(similar_imgs)):
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers+text',
                    marker=dict(size=12, color='green'),
                    text=[f"{label_dict[str(cls)]}" for cls in classes],
                    textposition="top center",  # Mantieni il testo in cima
                    textfont=dict(size=12, color="white")
                ))              

            
                img_base64 = image_to_base64(similar_imgs[i])  # Converte l'immagine in base64

                # Posiziona l'immagine nel grafico in base alle coordinate x, y
                fig.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{img_base64}",
                        xref="x",
                        yref="y",
                        x=x_values[i],
                        y=y_values[i],
                        sizex=(max(x_values) - min(x_values)) * 0.4,  
                        sizey=(max(y_values) - min(y_values)) * 0.4,  
                        opacity=0.8,
                        layer="above"
                    )   
                ) 


            fig.update_layout(
                hovermode="closest",
                hoverlabel=dict(bgcolor="black", font_size=13),
                showlegend=False,
                width = 500,
                xaxis=dict(scaleanchor="y",autorange=True),  
                yaxis=dict(autorange=True),
                #title="Hover over the points to see info."
            )

            st.plotly_chart(fig)

                #st.markdown("<br>", unsafe_allow_html=True) 

    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.write('Your case')
        st.image(st.session_state.uploaded_path, use_container_width=True)

    st.write('Your previous annotations:')
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes4}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes5}</h3>", unsafe_allow_html=True)

    if 'notes6' not in st.session_state:
        st.session_state.notes6 = "" 
    st.session_state.notes6 = st.text_area("Add notes. They'll be recorded.",
                                            value=st.session_state.notes6, height=100)
    selected_class = st.selectbox("Select a class:", options)


    cola, colb = st.columns([10, 1.5])
    with cola:
        if st.button("Back"):
            if 'log10' not in st.session_state:
                st.session_state.log10 = log_interaction("temporary_notes_6", st.session_state.notes6) 
            if 'log11' not in st.session_state:
                st.session_state.log11 = log_interaction(f'Fourth impression: {selected_class}')
            log_interaction("Going back...")  
            go_to_page(5)
            st.rerun()  
    with colb:
        proceedp6 = st.button('Proceed')
    if proceedp6:
        if selected_class != 'None':
            if 'log10' not in st.session_state:
                st.session_state.log10 = log_interaction("temporary_notes_6", st.session_state.notes6) 
            if 'log11' not in st.session_state:
                st.session_state.log11 = log_interaction(f'Fourth impression: {selected_class}')
            go_to_page(7)
            st.rerun()
        else:
            st.markdown("<p style='font-size: 1vw;'>You firstly need to select a class!</p>", unsafe_allow_html=True)


####    CONFIDENCES ###########################################
def pagina_7():
    if 'log10' in st.session_state:
        del st.session_state.log10
    if 'log11' in st.session_state:
        del st.session_state.log11

    st.title("Confidence")
    st.write("""Last step! You're observing the output of the neural network. 
    The histogram represents probabilities for each class. Hover on the bars
    to get more precise information. If you feel undecided, press 'I will choose later'.
             
    Warning: the model's accuracy is no 100% \and the model might be overconfident.""")
    col1, col2 = st.columns([0.9,0.3])
    
    with col1:
        i = load_image(image_path='./temp/case_study.jpg')
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(model(i.to(device)),dim=1)[0]
        probabilities = probabilities.cpu()
        print(probabilities)
        
        fig = go.Figure(data=[go.Bar(
            x=options[1:],  # Le etichette per l'asse X
            y=probabilities,  
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            opacity=0.75
        )])

        fig.update_layout(
            #title="Histogram of probabilities",
            yaxis_title="Probability",
            template="plotly_dark",  # Tema scuro
            plot_bgcolor="rgb(30, 30, 30)",  # Sfondo scuro
            paper_bgcolor="rgb(20, 20, 20)",  # Sfondo del grafico
            font=dict(color='white'),  # Colore del testo
            width = 500,
            xaxis_tickangle=-45,  # Angolo delle etichette sull'asse X
            yaxis=dict(range=[0,1])
        )

        st.plotly_chart(fig)
    
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.image(st.session_state.uploaded_path, use_container_width=True)

    st.write('Your previous annotations:')
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes3}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes4}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes5}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 0.8vw;'>{st.session_state.notes6}</h3>", unsafe_allow_html=True)

    if 'notes7' not in st.session_state:
        st.session_state.notes7 = "" 
    st.session_state.notes7 = st.text_area("Add notes. They'll be recorded.",
                                            value=st.session_state.notes7, height=100)
    
    selected_class = st.selectbox("Select a class:", [#'None',
                                                      'I will choose later',
                                                      'No_Pneumonia',
                                                      'Pneumonia'])
    
    cola, colb = st.columns([10, 1.5])
    with cola:  
        if st.button("Back"):
            if 'log12' not in st.session_state:
                st.session_state.log12 = log_interaction("temporary_notes_7", st.session_state.notes7) 
            if 'log13' not in st.session_state:
                st.session_state.log13 = log_interaction(f'Fifth impression: {selected_class}')
            log_interaction("Going back...")  
            go_to_page(6)
            st.rerun()  
    with colb:
        proceedp7 = st.button('Proceed')
    if proceedp7:
        if selected_class != 'None':

            if 'log12' not in st.session_state:
                st.session_state.log12 = log_interaction("temporary_notes_7", st.session_state.notes7) 
            if 'log13' not in st.session_state:
                st.session_state.log13 = log_interaction(f'Fifth impression: {selected_class}')

            if selected_class != 'I will choose later':
                dir_path = os.path.join(os.getcwd(),'./tuning/',class_dict[selected_class])
                if not os.path.exists(dir_path): 
                    try:
                        os.makedirs(dir_path)
                        #st.success(f"Directory './tuning/{selected_class}' created.")
                    except Exception as e:
                        st.error(f"Error creating directory: {str(e)}")

                files = os.listdir(dir_path)
                jpg_files = [f for f in files if f.endswith('.jpg') and f.split('.')[0].isdigit()]

                if jpg_files:
                    max_num = max(int(f.split('.')[0]) for f in jpg_files)
                    next_file = f"{max_num + 1}.jpg"
                else:
                    next_file = "0.jpg"

                with open(os.path.join(dir_path,next_file), "wb") as f:
                    f.write(get_image_bytes(st.session_state.uploaded_path)) 

                augment_case_study(img_path=os.path.join(dir_path,next_file),
                                   temporary_folder='./temporary_images',class_to_exclude=class_dict[selected_class])

                go_to_page(10)
                st.rerun()
            else:
                go_to_page(10)
                st.rerun()
        else:
            st.markdown("<p style='font-size: 1vw;'>You firstly need to select a class!</p>", unsafe_allow_html=True)


####    END SESSION #####################################
def pagina_10():

    if 'notes6' in st.session_state:
        backpage = 7
    else:
        backpage = 1
    if 'log12' in st.session_state:
        del st.session_state.log12
    if 'log13' in st.session_state:
        del st.session_state.log13

    st.title('Session ended')
    st.write('Click on "Download" to receive the csv file of your interactions. Double click on "New session" to begin a new session.')

    
    numero_file = sum(len(files) for _, _, files in os.walk('./tuning'))
    if numero_file >6:   
        st.write('You have reached a sufficiently wide tuning set. Now, wait: the model is updating...')
        with st.spinner():
            import fine_tuning
            st.success("Tuned successfully!")
            log_interaction('Model parameters have been tuned.')
    else:   
        st.write(f'Up until now you have collected a tuning set of {numero_file} images. When you will have collected more than 6 new cases, a finetunig will automatically be carried out.')
   

    cola, colb, colc = st.columns([2,3.1,1])
    with cola:
        if st.button("Back"):
            log_interaction("Going back...")  
            go_to_page(backpage)    
            st.rerun()   
    with colb:

        try:
            with open('user_interactions.csv', mode='r', newline='') as file:
                f = file.read()  # Leggi il contenuto del file CSV

            # Mostra il pulsante per scaricare il file CSV
            st.download_button(
                label="Download", 
                data=f,  # Passa il contenuto del file come stringa
                file_name="your_interactions.csv", 
                mime="text/csv"
            )

        except FileNotFoundError:
            st.warning("No interaction log found. Are you sure it exists?")
    with colc:
        if st.button("Exit"):
                st.stop() 
    
    if st.button("New session"):
        log_interaction('Beginning new session') 
        if 'gradcam' in st.session_state:
            del st.session_state.gradcam
        if 'rise' in st.session_state:
            del st.session_state.rise
        go_to_page(2)


              



 

rise_N = 1500
options = ['None','No_Pneumonia','Pneumonia']
class_dict = {'No_Pneumonia': '0', 
                       'Pneumonia': '1'}
label_dict = {'0': 'No_Pneumonia',
                '1': 'Pneumonia'} 


if "pagina" not in st.session_state:
    print('Adding page session state...')
    st.session_state.pagina = 1 

# Logica per visualizzare la pagina corretta in base al valore di st.session_state.pagina
if st.session_state.pagina == 1:
    pagina_1()
elif st.session_state.pagina == 2: 
    pagina_2()
elif st.session_state.pagina == 3:
    pagina_3()
elif st.session_state.pagina == 4:
    pagina_4() 
elif st.session_state.pagina == 5:
    pagina_5()      
elif st.session_state.pagina == 6:
    pagina_6()
elif st.session_state.pagina == 7:
    pagina_7()
elif st.session_state.pagina == 10:
    pagina_10()
