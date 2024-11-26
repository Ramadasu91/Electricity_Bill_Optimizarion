import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from geneticalgorithm import geneticalgorithm as ga

# Define the fitness function for optimization
def fitness_function(x):
    # Scale the inputs
    x_scaled = scaler.transform([x])
    # Predict power consumption
    prediction = model.predict(x_scaled)
    return prediction[0]  # Minimize predicted power consumption

# Main Streamlit application
def main():
    st.title("Power Consumption Optimization - Cement Industry")

    st.write("""
    This application allows you to upload a dataset, optimize the power consumption,
    and generate a new dataset with optimized feature values.
    """)

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file:
        # Read uploaded dataset
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.write(data.head())

        # Validate required columns
        required_columns = [
            "Clinker Production Rate (tons/hr)", "Kiln Temp (°C)", "Grinding Mill Load (kWh)",
            "Fuel Type", "Raw Material Hardness", "Ambient Temp (°C)", "Humidity (%)",
            "Cement Grade", "Renewable Energy Usage (%)", "Production Volume (tons/day)",
            "Maintenance Downtime (hrs)", "Power Consumed (kWh)"
        ]
        if not all(col in data.columns for col in required_columns):
            st.error("The uploaded dataset must contain all required columns.")
            return

        # Preprocessing: Encode categorical variables and scale numerical features
        data = data.copy()
        label_encoders = {}
        for col in ["Fuel Type", "Cement Grade"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        features = data.drop(columns=["Power Consumed (kWh)"])
        target = data["Power Consumed (kWh)"]

        global scaler, model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Train Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_scaled, target)

        # Genetic Algorithm Parameters
        variable_boundaries = np.array([
            [100, 150],   # Clinker Production Rate (tons/hr)
            [1300, 1600], # Kiln Temp (°C)
            [400, 600],   # Grinding Mill Load (kWh)
            [0, 2],       # Fuel Type (Encoded: 0, 1, 2)
            [6.0, 9.0],   # Raw Material Hardness
            [30, 40],     # Ambient Temp (°C)
            [50, 70],     # Humidity (%)
            [0, 2],       # Cement Grade (Encoded: 0, 1, 2)
            [5, 20],      # Renewable Energy Usage (%)
            [2000, 3000], # Production Volume (tons/day)
            [0, 10],      # Maintenance Downtime (hrs)
        ])

        algorithm_params = {
            'max_num_iteration': 100,
            'population_size': 50,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        # Run Genetic Algorithm Optimization
        st.write("Optimizing... This may take a few moments.")
        model_ga = ga(function=fitness_function,
                      dimension=11,
                      variable_type='real',
                      variable_boundaries=variable_boundaries,
                      algorithm_parameters=algorithm_params)

        model_ga.run()
        optimized_solution = model_ga.output_dict['variable']
        predicted_min_power = model_ga.output_dict['function']

        # Generate optimized dataset
        optimized_data = data.copy()
        optimized_data.iloc[:, :-1] = scaler.inverse_transform(np.tile(optimized_solution, (data.shape[0], 1)))
        optimized_data["Power Consumed (kWh)"] = predicted_min_power

        st.write("Optimized Dataset (First 5 Rows):")
        st.write(optimized_data.head())

        # Save optimized dataset to CSV
        optimized_file_path = "optimized_cement_industry_data.csv"
        optimized_data.to_csv(optimized_file_path, index=False)

        # Provide download link
        st.download_button(
            label="Download Optimized Dataset",
            data=optimized_data.to_csv(index=False),
            file_name=optimized_file_path,
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
