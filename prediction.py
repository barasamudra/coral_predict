import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load model

model = load_model('model.keras')



def run():

    # Add title
    st.title("Coral Bleach Prediction")

    # Add image
    #img = Image.open('jantung.png')
    #st.image(img)

    def prediction(file, img_height=200, img_width=225):
        
        ## Load an image
        image_inf = tf.keras.utils.load_img(file, target_size=(img_height, img_width))

        ## Rescaling and reshaping image
        x = tf.keras.utils.img_to_array(image_inf)/255
        x = np.expand_dims(x, axis=0)
        image_infs = np.vstack([x])

         ## Predict
        y_pred_inf_proba = model.predict(image_infs, batch_size=10)
        y_pred_inf_cls = np.argmax(y_pred_inf_proba)
        class_names = ['bleached_corals', 'healthy_corals']
        y_pred_class_name = class_names[np.argmax(y_pred_inf_proba[0])]

        return image_inf, y_pred_class_name
    
    with st.form("upload_image"):
        data_inf = st.file_uploader('upload image:')
        submit = st.form_submit_button('Predict')



    if submit:
        # Predict using model
        y_pred_inf, y_pred_class_name = prediction(data_inf)
        if y_pred_class_name == 'bleached_corals':
           result = "This coral is bleached"
        else:
           result = "This coral is healthy"

        st.image(y_pred_inf)
        st.write(result)

if __name__ == '__main__':
  run()