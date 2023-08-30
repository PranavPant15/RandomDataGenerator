import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt  # Import Matplotlib
from ctgan import CTGAN
from table_evaluator import TableEvaluator

# Set Matplotlib backend
matplotlib.use('Agg')

# Load the CSV data
data = pd.read_csv('./insurance.csv')

# Define categorical features
categorical_features = ['age', 'sex', 'children', 'smoker', 'region']

# Create a Streamlit app
def main():
    st.title('CTGAN Streamlit App')

    # Display the original data
    st.subheader('Original Data')
    st.write(data)

    # Train CTGAN model and generate samples
    ctgan = CTGAN(verbose=False)
    ctgan.fit(data, categorical_features, epochs=200)
    samples = ctgan.sample(1000)

    # Display generated samples
    st.subheader('Generated Samples')
    st.write(samples)

    # Perform visual evaluation using TableEvaluator
    table_evaluator = TableEvaluator(data, samples, cat_cols=categorical_features)

    # Display visual evaluation results
    st.subheader('Visual Evaluation')
    
    # Generate the evaluation plot
    evaluation_plot = table_evaluator.visual_evaluation()

    # Display the evaluation plot using Streamlit's magic command
    st.pyplot(evaluation_plot)

if __name__ == '__main__':
    main()

