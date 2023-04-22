import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Set up the Streamlit app layout
st.set_page_config(page_title='Regression Analysis App')
st.title('Regression Analysis App')
st.write(
    "Take any dataset as input and specify the target column. Display a visual representation of the linear regression line with selectable features from a dropdown menu. Implement MLP for regression (can have features as input, optional). Provide dropdown menus for various data normalization techniques and weight initialization methods for MLP. Evaluate the performance of different normalization techniques by comparing their scores."
) 

# # Upload the dataset
# file = st.file_uploader('Upload a CSV file', type='csv')

st.sidebar.write("Create random dataset")

# Create a random dataset
n_samples = st.sidebar.number_input('Number of samples', min_value=1, max_value=1000, value=15)
n_features = st.sidebar.number_input('Number of features', min_value=1, max_value=100, value=5)
X = np.random.randn(n_samples, n_features)
y = np.sum(X, axis=1) + np.random.randn(n_samples)

st.sidebar.write("---")

# Convert to a pandas dataframe
data = pd.DataFrame(np.column_stack([X, y]), columns=["target"] + [f"feature_{i}" for i in range(n_features)])

if True:
    df = data 

    st.sidebar.write("Linear regression")

    # Select the target column
    target_col = st.sidebar.selectbox('Select the target column', options=list(df.columns))

    # Select the features for the linear regression line
    lr_features = st.sidebar.multiselect('Select features for linear regression', options=list(df.columns))

    # Plot the linear regression line
    if lr_features:
        X_lr = df[lr_features].values
        y_lr = df[target_col].values
        lr = LinearRegression().fit(X_lr, y_lr) 
        y_lr_pred = lr.predict(X_lr) 

        fig, ax = plt.subplots() 
        ax.scatter(y_lr, y_lr_pred, label="Data")
        ax.plot([y_lr.min(), y_lr.max()], [y_lr.min(), y_lr.max()], color='red', label="Predicted")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

    st.sidebar.write("---")

    st.sidebar.write("MLP regression")

    # Select the features for the MLP regression
    mlp_features = st.sidebar.multiselect('Select features for MLP regression', options=list(df.columns))

    # Specify the MLP parameters
    mlp_hidden_layer_sizes = st.sidebar.slider('Hidden layer sizes', min_value=1, max_value=100, value=(10, 10))
    mlp_activation = st.sidebar.selectbox('Activation function', options=['identity', 'logistic', 'tanh', 'relu'])
    mlp_max_iter = st.sidebar.number_input('Max iterations', min_value=1, max_value=1000, value=200)

    st.sidebar.write("---")

    # Select the data normalization technique
    normalization = st.sidebar.selectbox('Select data normalization technique',
                                 options=['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler'])

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
    elif weight_init == 'Glorot Uniform':
        mlp = MLPRegressor(hidden_layer_sizes=mlp_hidden_layer_sizes,activation=mlp_activation,max_iter=mlp_max_iter,random_state=42,weight_init='glorot_uniform')
    elif weight_init == 'He Uniform':
        mlp = MLPRegressor(hidden_layer_sizes=mlp_hidden_layer_sizes,activation=mlp_activation,max_iter=mlp_max_iter,random_state=42,weight_init='he_uniform')

    # Fit the MLP model and make predictions
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    st.write("---")

    # Evaluate the performance of the MLP model
    st.write("**Evaluating the performance of the MLP model:**")
    st.write("Data Normalization Method = " + normalization + " | Weight Initialization Method =" + weight_init)
    st.write('- MLP R-squared:', r2_score(y_test, y_pred))
    st.write('- MLP MSE:', mean_squared_error(y_test, y_pred))

    st.write("---")

    # Compare the performance of different normalization techniques
    st.write("**Comparing the performance of different normalization techniques:**")
    scaler_options = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
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
        st.write("- " + scaler_option, 'MLP R-squared:', r2_score(y_test, y_pred_scaled))
        st.write("- " + scaler_option, 'MLP MSE:', mean_squared_error(y_test, y_pred_scaled))
