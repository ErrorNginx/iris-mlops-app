import streamlit as st
import mlflow.pytorch
import torch
import pandas as pd

# Load the model from MLflow Model Registry using tag
def load_model(model_name, tag):
    client = mlflow.tracking.MlflowClient()
    model_info = client.search_model_versions(f"name='{model_name}' and tags.stage='{tag}'")[0]
    model_uri = f"models:/{model_info.name}/{model_info.version}"
    model = mlflow.pytorch.load_model(model_uri)
    return model

model_name = "IrisModel"
tag = "Production"
model = load_model(model_name, tag)

# Streamlit app
st.title('Iris Species Prediction')
st.write('This application predicts the species of an iris flower based on its sepal and petal measurements. Use the sliders in the sidebar to input the measurements.')

# Sidebar sliders for user input
st.sidebar.header('Input Parameters')
st.sidebar.write('Adjust the sliders to input the measurements of the iris flower.')

sepal_length = st.sidebar.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 0.0, 10.0, 3.5)
petal_length = st.sidebar.slider('Petal Length (cm)', 0.0, 10.0, 1.5)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.0, 10.0, 0.2)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'sepal_length': [sepal_length],
    'sepal_width': [sepal_width],
    'petal_length': [petal_length],
    'petal_width': [petal_width]
})

# Convert the input data to a PyTorch tensor
input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

# Display input data
st.write('### Input Measurements')
st.write(input_data)

# Make the prediction
if st.sidebar.button('Predict'):
    with torch.no_grad():
        model.eval()
        prediction = model(input_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Map numerical predictions to species names
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_map[predicted_class]

    # Display the prediction
    st.write('### Prediction')
    st.write(f'The predicted species is: **{predicted_species}**')
    
    1

    # Display a message based on the prediction
    if predicted_species == 'Iris-setosa':
        st.image('images/Iris-setosa.jpg', width=300)
        st.write('**Iris setosa** is an iris species commonly found in the wild.')
    elif predicted_species == 'Iris-versicolor':
        st.image('images/Iris-versicolor.jpg', width=300)
        st.write('**Iris versicolor**, also known as the Blue Flag iris, is a species native to North America.')
    elif predicted_species == 'Iris-virginica':
        st.image('images/Iris-virginica.jpg', width=300)
        st.write('**Iris virginica** is a species commonly found in wetlands of the eastern United States.')
# Sidebar footer
st.sidebar.write('---')
st.sidebar.write('Developed by [Your Name](https://your-website.com)')
