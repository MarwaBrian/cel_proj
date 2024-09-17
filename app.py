import numpy as np
import pandas as pd
import streamlit as st
import joblib



#deploy on streamlit
reg = joblib.load('./match_pred.pkl')


# Title and description
st.title('How many goals will be scored in a match?')
st.write("Input feature values to use for the prediction.")


#input field for each feature
feature_1 = st.number_input('Home Team Points Per Game (Pre-Match)(min=0.1, max=3.0)', min_value=0.1, max_value=3.0)  
feature_2 = st.number_input('Away Team Points Per Game (Pre-Match)(min=0.1, max=3.0)', min_value=0.1, max_value=3.0)
feature_3 = st.number_input('Home Team Points Per Game (Current)(min=0.1, max=3.0)', min_value=0.1, max_value=3.0)
feature_4 = st.number_input('Away Team Points Per Game (Current)(min=0.1, max=3.0)', min_value=0.1, max_value=3.0)
feature_5 = st.number_input('Average Goals(min=0.8, max=7)', min_value=0.8, max_value=7.0)
feature_6 = st.number_input('BTTS Average(min=12, max=100)', min_value=12, max_value=100, step=1)
feature_7 = st.number_input('Over05 Average(min=50, max=100)', min_value=50, max_value=100, step=1)
feature_8 = st.number_input('Over15 Average(min=17, max=100)', min_value=17, max_value=100, step=1)
feature_9 = st.number_input('Home Team Pre-Match xG(min=0.0, max=4.2)', min_value=0.0, max_value=4.2)
feature_10 = st.number_input('Away Team Pre-Match xG(min=0.0, max=3.0)', min_value=0.0, max_value=3.1)
feature_11 = st.number_input('Game Week(min=1, max=34)', min_value=1, max_value=34, step=1)
feature_12 = st.number_input('Result - Home Team Goals(min=0, max=15)', min_value=0, max_value=15, step=1)
feature_13 = st.number_input('Result - Away Team Goals(min=0, max=15)', min_value=0, max_value=15, step=1)

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