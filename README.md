# Regression Analysis App 

## Collaborators 
Saatvik Rao 

R Yesshu Dhurandhar 

## Project Goal  
Take any dataset as input and specify the target column. Display a visual representation of the linear regression line with selectable features from a dropdown menu. Implement MLP for regression (can have features as input, optional). Provide dropdown menus for various data normalization techniques and weight initialization methods for MLP. Evaluate the performance of different normalization techniques by comparing their scores. 

## Instructions 
### To use in localhost 
- In your terminal, go to directory containing `streamlit_app.py`.
- Run the following command in the terminal: `streamlit run streamlit_app.py`. 

### Hosted 
- To use through the hosted website go to the following URL: 

    [Link to Streamlit App](https://saatvikraoiitgn-yeeshu-saatvik-assignment3-streamlit-app-ycq22a.streamlit.app) 


## Customizations 
- You can input a csv file for data, just uncomment the code. Also, you can take random data by giving input of the number of samples and features. 
- Linear Regression:
    - Choose the target feature 
    - Choose one of the features for linear regression line
- MLP Regression:
    - Choose one of the features for MLP regression 
- Data normalization: You can choose from three data normalization techniques: 
    - StandardScaler
    - MinMaxScaler
    - RobustScalar
- Weight initialization: You can choose from three weight initialization techniques: 
    - Random
    - Glorot Uniform
    - He Uniform 
