# Reverse_Image_Search
<b>Problem statement:<b>
A reverse image search engine based on the fundamental Deep Learning algorithm, viz., K-NN algorithm(K-Nearest Neighbours). 


Exploratory Data Analysis (EDA) and Data preparation:
As for data collection, we used a publicly available dataset sourced from kaggle which contains ~4700 images to train our model and used the scikit.learn library to extract and process features out of the query(the image uploaded by the user) image.


Model Training and Evaluation:
Post data collection, we trained the model using the standard ResNet 50 function which is a 50 layers deep convolution neural network and was able to provide us accuracy close to ~81%. The algorithm is defined such that even the previously uploaded 'query' images are queued to be trained through very same process, thus making the model much more robust, efficient and accurate with each uploaded 'query' image.


Model Deployment and monitoring:
For deployment, we used a python library called 'Streamlit' by installing it into our operating system through "pip install streamlit". Using it only requires us to define the path of the file that we intend to run in the Command Line Prompt byy using a simple command, "streamlit run filename.py" and it locally deploys the application itself. 
It is an open-source python framework for building web applications for Machine Learning and Data Science. We can instantly develop web applications and deploy them easily using Streamlit. It allows us to write an application the same way we write a python code. It makes it seamless to work on the interactive loop of coding and viewing results in the web application. The availability of HTML markdown in Streamlit helps in peronalizing the user interface as per requirements.
