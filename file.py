import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
# @st.cache
# def load_data():
#     data = pd.read_csv("path_to_your_dataset.csv") 
#     return data

# data = load_data()

# Create a random dataset
n_samples = 10
n_features = 5
X = np.random.randn(n_samples, n_features)
y = np.sum(X, axis=1) + np.random.randn(n_samples)

# Convert to a pandas dataframe
data = pd.DataFrame(np.column_stack([X, y]), columns=[f"feature_{i}" for i in range(n_features)] + ["target"])

# Create a streamlit app
st.title("Linear Regression")
st.write(
    "Take any dataset as input and specify the target column. Display a visual representation of the linear regression line with selectable features from a dropdown menu. Implement MLP for regression (can have features as input, optional). Provide dropdown menus for various data normalization techniques and weight initialization methods for MLP. Evaluate the performance of different normalization techniques by comparing their scores."
)

# Specify the target column
target_col = st.sidebar.selectbox("Select target column", data.columns)

# Choose features from a dropdown menu 
features = st.sidebar.multiselect("Select features", data.columns.drop(target_col))

# Create a linear regression model
lm = LinearRegression()

# Fit the model
lm.fit(data[features], data[target_col])

# Create a scatter plot of the data and the regression line
fig, ax = plt.subplots()
ax.scatter(data[features], data[target_col], alpha=0.5)
ax.plot(data[features], lm.predict(data[features]), color='red')
ax.set_xlabel("Features")
ax.set_ylabel("Target")
st.pyplot(fig)

# Create a MLP regressor
mlp = MLPRegressor(random_state=0)

# Choose normalization techniques and weight initialization methods from dropdown menus
normalization = st.sidebar.selectbox("Select normalization technique", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
if normalization == "StandardScaler":
    scaler = StandardScaler()
elif normalization == "MinMaxScaler":
    scaler = MinMaxScaler()
else:
    scaler = None

weight_init = st.sidebar.selectbox("Select weight initialization method", ["Random", "Normal", "He Uniform", "He Normal", "Glorot Uniform", "Glorot Normal"])
if weight_init == "Random":
    mlp.set_params(random_state=0)
elif weight_init == "Normal":
    mlp.set_params(random_state=0, initialization="normal")
elif weight_init == "He Uniform":
    mlp.set_params(random_state=0, initialization="he_uniform")
elif weight_init == "He Normal":
    mlp.set_params(random_state=0, initialization="he_normal")
elif weight_init == "Glorot Uniform":
    mlp.set_params(random_state=0, initialization="glorot_uniform")
elif weight_init == "Glorot Normal":
    mlp.set_params(random_state=0, initialization="glorot_normal")

# Normalize the data
if scaler is not None:
    data[features] = scaler.fit_transform(data[features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_col], test_size=0.2, random_state=0)

# Fit the MLP regressor
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Calculate the R-squared score
score = r2_score(y_test, y_pred)

# Display the score
st.write(f"R-squared score: {score:.4f}") 