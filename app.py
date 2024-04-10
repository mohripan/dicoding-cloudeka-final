import streamlit as st
import tensorflow as tf
import requests
import json
import base64
from pprint import PrettyPrinter

# Function to create a tf.Example message
def create_tf_example(features):
    feature = {}
    for key, value in features.items():
        if isinstance(value, int):
            feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, str):
            value = bytes(value, 'utf-8')
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Example(features=tf.train.Features(feature=feature))

# UI for feature input
st.title('Obesity Level Prediction')

# Categorical features with their options
categories = {
    'CAEC': ['Always', 'Frequently', 'Sometimes', 'no'],
    'CALC': ['Always', 'Frequently', 'Sometimes', 'no'],
    'FAVC': ['no', 'yes'],
    'Gender': ['Female', 'Male'],
    'MTRANS': ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'],
    'SCC': ['no', 'yes'],
    'SMOKE': ['no', 'yes'],
    'family_history_with_overweight': ['no', 'yes']
}

# Input fields for categorical features
features = {cat: st.selectbox(cat, options) for cat, options in categories.items()}

# Input fields for numerical features
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for feature in numerical_features:
    features[feature] = st.number_input(f'{feature} (float)', format="%.2f")

# Prediction button
if st.button('Predict Obesity Level'):
    tf_example = create_tf_example(features)
    serialized_example = tf_example.SerializeToString()
    
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": [{"b64": base64.b64encode(serialized_example).decode('utf-8')}]
    })
    
    # TensorFlow Serving URL
    MODEL_URL = 'http://103.190.215.250:8501/v1/models/ob-model:predict'
    
    # Sending the request
    response = requests.post(MODEL_URL, data=data, headers={"Content-Type": "application/json"})
    
    # Processing the response
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        st.write(predictions)
    else:
        st.error("Error during model inference:", response.text)
    
    pp = PrettyPrinter()
    pp.pprint(response.json())
    
    LABEL_MAPPING = {
        0: 'Normal_Weight',
        1: 'Overweight_Level_I',
        2: 'Overweight_Level_II',
        3: 'Obesity_Type_I',
        4: 'Insufficient_Weight',
        5: 'Obesity_Type_II',
        6: 'Obesity_Type_III'
    }
    
    predictions = response.json()['predictions'][0]
    max_index = predictions.index(max(predictions))
    
    st.write("Prediction probabilities:")
    for index, probability in enumerate(predictions):
        st.write(f"{LABEL_MAPPING[index]}: {probability:.4f}")
    
    st.write(f"Class with the highest probability: {LABEL_MAPPING[max_index]}")
