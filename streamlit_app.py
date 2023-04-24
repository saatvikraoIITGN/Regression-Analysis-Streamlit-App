import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression 

# Set up the Streamlit app layout
st.set_page_config(page_title='Regression Analysis App')
st.title('Regression Analysis App')
st.write(
    "Take any dataset as input and specify the target column. Display a visual representation of the linear regression line with selectable features from a dropdown menu. Implement MLP for regression (can have features as input, optional). Provide dropdown menus for various data normalization techniques and weight initialization methods for MLP. Evaluate the performance of different normalization techniques by comparing their scores."
) 

st.write("---")

# # Upload the dataset
# file = st.file_uploader('Upload a CSV file', type='csv')

st.sidebar.write("Create random dataset")

# Create a random dataset
n_samples = st.sidebar.number_input('Number of samples', min_value=1, max_value=1000, value=15)
n_features = st.sidebar.number_input('Number of features', min_value=1, max_value=100, value=5)

X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
X = np.interp(X, (X.min(), X.max()), (-10, 10))

st.sidebar.write("---")

# Convert to a pandas dataframe
data = pd.DataFrame(np.column_stack([X, y]), columns=["target"] + [f"feature_{i}" for i in range(n_features)])

df = data 

st.write("**Linear Regression:**")

st.sidebar.write("Linear regression")

# Select the target column
target_col = st.sidebar.selectbox('Select the target column', options=list(df.columns)) 

# Select one of the features for the linear regression line
lr_features = st.sidebar.selectbox('Select a feature for the linear regression line', options=list(df.columns))
# convert lr_features to a list
lr_features = [lr_features] 

# Plot the linear regression line
if lr_features:
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(df[lr_features].values, df[target_col].values, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_lr_train, y_lr_train)
    y_lr_pred = lr.predict(X_lr_test)

    fig, ax = plt.subplots()
    ax.scatter(X_lr_train, y_lr_train, label="Data") 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Linear Regression')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_ext = np.linspace(xlim[0], xlim[1], 100)
    p = np.polyfit(X_lr_train.flatten(), y_lr_train, deg=1)
    y_ext = np.poly1d(p)(x_ext)
    ax.plot(x_ext, y_ext, color='red', label="Predicted") 
    fig.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)



st.sidebar.write("---")

st.sidebar.write("MLP Regression")

st.write("**MLP regression:**")

# Select the features for the MLP regression
mlp_features = st.sidebar.selectbox('Select features for MLP regression', options=list(df.columns))
mlp_features = [mlp_features]

# Specify the MLP parameters
mlp_hidden_layer_sizes = st.sidebar.slider('Hidden layer sizes', min_value=1, max_value=100, value=(10, 10))
mlp_activation = st.sidebar.selectbox('Activation function', options=['identity', 'logistic', 'tanh', 'relu'])
mlp_max_iter = st.sidebar.number_input('Max iterations', min_value=1, max_value=1000, value=200)

st.sidebar.write("---")

# Select the data normalization technique
normalization = st.sidebar.selectbox('Select data normalization technique', options=['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler'])

# Select the weight initialization method
weight_init = st.sidebar.selectbox('Select weight initialization method',options=['Random', 'Glorot', 'He'])

# Preprocess the data
X = df[mlp_features].values
y = df[target_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if normalization == 'StandardScaler':
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
elif normalization == 'MinMaxScaler':
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
elif normalization == 'RobustScaler':
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Initialize the MLP model
if weight_init == 'Random':
    mlp = MLPRegressor(hidden_layer_sizes=mlp_hidden_layer_sizes,activation=mlp_activation,max_iter=mlp_max_iter,random_state=42)
elif weight_init == 'Glorot':
    mlp = MLPRegressor(hidden_layer_sizes=mlp_hidden_layer_sizes,activation=mlp_activation,max_iter=mlp_max_iter,random_state=42,solver='adam')
elif weight_init == 'He':
    mlp = MLPRegressor(hidden_layer_sizes=mlp_hidden_layer_sizes,activation=mlp_activation,max_iter=mlp_max_iter,random_state=42,solver='adam')

# Fit the MLP model and make predictions
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

fig, ax = plt.subplots() 
ax.scatter(X_train, y_train, label="Data")
ax.set_xlabel('X') 
ax.set_ylabel('Y')
ax.set_ylim(-10, 10) 
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_ext = np.linspace(xlim[0], xlim[1], 100)
p = np.polyfit(X_train.flatten(), y_train, deg=1)
y_ext = np.poly1d(p)(x_ext)
ax.plot(x_ext, y_ext, color='red', label="Predicted")
ax.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig) 

st.write("---")

# Evaluate the performance of the MLP model
st.write("**Evaluating the performance of the MLP model:**")
st.write("Data Normalization Method = " + normalization + " | Weight Initialization Method =" + weight_init)
st.write('- MLP R-squared:', r2_score(y_test, y_pred))
st.write('- MLP MSE:', mean_squared_error(y_test, y_pred))

st.write("---")

# Comparing the performance of different normalization techniques
st.write("**Comparing the performance of different normalization techniques:**")

scaler_options = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
results = [] 

for scaler_option in scaler_options:
    if scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_option == 'RobustScaler':
        scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlp.fit(X_train_scaled, y_train)
    y_pred_scaled = mlp.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_scaled)
    mse = mean_squared_error(y_test, y_pred_scaled) 
    results.append([scaler_option, r2, mse]) 

results_df = pd.DataFrame(results, columns=['Normalization technique', 'R-squared', 'MSE'])
st.write(results_df)