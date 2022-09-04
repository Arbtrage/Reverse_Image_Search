import streamlit as st
import os
import keras
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

f_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])



st.title('Reverse Image Search ')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('upload',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0




def feature_extraction(img_path,model):
    img =  keras.utils.load_img(img_path, target_size=(224, 224))
    img_array =  keras.utils.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


#Returns the indices of the recommended pictures
def recommend(features,f_list):
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    neighbors.fit(f_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):

        # display the file
        display_image = Image.open(uploaded_file)
        st.title('Query:')
        st.image(display_image)

        # feature extraction
        features = feature_extraction(os.path.join("upload",uploaded_file.name),model)
        
        indices = recommend(features,f_list)
        st.title('Result:')
        # Display the similar images
        col1,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)

        
        with col1:
            st.image(filenames[indices[0][0]], width=130)
            st.image(filenames[indices[0][1]], width=130)
        with col3:
            st.image(filenames[indices[0][2]], width=130)
            st.image(filenames[indices[0][3]], width=130)
        with col5:
            st.image(filenames[indices[0][4]], width=130)
            st.image(filenames[indices[0][5]], width=130)
        with col7:
            st.image(filenames[indices[0][6]], width=130)
            st.image(filenames[indices[0][7]], width=130)
        with col9:
            st.image(filenames[indices[0][8]], width=130)
            st.image(filenames[indices[0][9]], width=130)
    else:
        st.header("Please upload the file again")

