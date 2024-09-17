import numpy as np
import pandas as pd
import streamlit as st
import joblib



#deploy on streamlit
reg = joblib.load('./match_pred.pkl')


# Title and description
st.title('Linear Regression Model')
st.write("Input feature values to test the model.")


#input field for each feature
feature_1 = st.number_input('Home Team Points Per Game (Pre-Match)')  
feature_2 = st.number_input('Away Team Points Per Game (Pre-Match)')
feature_3 = st.number_input('Home Team Points Per Game (Current)')
feature_4 = st.number_input('Away Team Points Per Game (Current)')
feature_5 = st.number_input('Average Goals')
feature_6 = st.number_input('BTTS Average')
feature_7 = st.number_input('Over05 Average')
feature_8 = st.number_input('Over15 Average')
feature_9 = st.number_input('Home Team Pre-Match xG')
feature_10 = st.number_input('Away Team Pre-Match xG')
feature_11 = st.number_input('Game Week')
feature_12 = st.number_input('Result - Home Team Goals')
feature_13 = st.number_input('Result - Away Team Goals')

#when the user clicks the "Predict" button
if st.button('Predict'):
    #make predictions based on input values
    input_features = np.array([[
        feature_1,
        feature_2,
        feature_3,
        feature_4,
        feature_5,
        feature_6,
        feature_7,
        feature_8,
        feature_9,
        feature_10,
        feature_11,
        feature_12,
        feature_13,
    ]])
    prediction = reg.predict(input_features)
    
    st.write(f"Predicted Value: {prediction[0]}")