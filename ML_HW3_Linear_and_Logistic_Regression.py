#=================================================
#   ML_HW3_Linear_and_Logistic_Regression
#   Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
#   Using Python 3.9.16 & Spyder IDE
#=================================================

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#%% Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from textwrap import wrap


#%% Pre-processing the data
""" a) Pre-processing the data: To implement the regression algorithm, you 
must first prepare the data."""

# Load dataset
housing = pd.read_csv('./housing.csv')


# Check data
housing
housing.info() # get a quick description of the data
housing.describe() # shows a summary of the numerical attributes


# Missing values
print(housing.isnull().sum()) #check missing data
housing.fillna(housing.mean(), inplace=True) # Replace missing values with mean


# Duplicated data
print(housing.duplicated().sum()) # check duplicated data


# Scaling data | Scale numerical features to 0-1
scaler = MinMaxScaler()
housing[housing.select_dtypes(include=['float64']).columns] = scaler.fit_transform(housing[housing.select_dtypes(include=['float64']).columns])


# Dummy variable indicator method | Convert non-numerical values into numbers
housing["ocean_proximity"].value_counts() # find out what categories exist and how many districts belong to each category
housing = pd.get_dummies(housing)

# Check data again
housing
housing.info() # get a quick description of the data
housing.describe() # shows a summary of the numerical attributes


# Visualization
Fig1 = housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                    s=housing["population"], label='population',
                    c='median_house_value', cmap='jet', colorbar=True,
                    legend=True, sharex=False, figsize=(10,7))



#%% Split the data into features and labels
""" b) Split the data randomly into three parts, 60% - 20% - 20% respectively 
for training, validation and test set."""

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

# Split the data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)



#%% Linear Regularization Algorithm
"""c) Run the linear regularization algorithm with regularization and early 
stopping as follows: 
• For 5 learning rate 0.1, 0.3, 0.5, 0.7 and 1. 
• For 4 regularization coefficients 0, 0.1, 1, 10. 
• Plot the loss curves for the training and validation data in one plot.
(Totally 20 plots.)"""


# Define function to train and evaluate model
def train_and_evaluate_model(alpha, learning_rate):
    model = SGDRegressor(alpha=alpha, learning_rate='constant', eta0=learning_rate, early_stopping=True, validation_fraction=0.1)
    history = model.fit(X_train, y_train)
    train_loss = mean_squared_error(y_train, model.predict(X_train))
    val_loss = mean_squared_error(y_val, model.predict(X_val))
    test_loss = mean_squared_error(y_test, model.predict(X_test))
    return history.coef_, train_loss, val_loss, test_loss

# Define learning rates and regularization coefficients
learning_rates = [0.1, 0.3, 0.5, 0.7, 1]
reg_coefs = [0, 0.1, 1, 10]
TrainLoss = []
ValLoss = []
TestLoss = []

# Train and evaluate models for each combination of learning rate and regularization coefficient
for learning_rate in learning_rates:
    for reg_coef in reg_coefs:
        
        # Train and evaluate model
        coef_, train_loss, val_loss, test_loss = train_and_evaluate_model(reg_coef, learning_rate)
        
        TrainLoss.append(train_loss)
        ValLoss.append(val_loss)
        TestLoss.append(test_loss)
        
        # Plot loss curves for training and validation data
        plt.plot(coef_)
        plt.title("\n".join(wrap(f'Learning rate: {learning_rate}, Regularization coefficient: {reg_coef}, Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}')))
        plt.ylabel('Coefficient')
        plt.xlabel('Feature')
        plt.show()



#%%
""" d) Finally, choose the best parameters and report the cost value in the 
test data with these parameters."""

Best_learning_rate = learning_rates[np.argmin(train_loss)]
Best_reg_coef = reg_coefs[np.argmin(train_loss)]
min_train_loss = TrainLoss[np.argmin(train_loss)]
min_val_loss = ValLoss[np.argmin(val_loss)]
min_test_loss = TestLoss[np.argmin(test_loss)]
print("Best_learning_rate:", Best_learning_rate)
print("Best_reg_coef:", Best_reg_coef)
print("min_train_loss:", min_train_loss)
print("min_val_loss:", min_train_loss)
print("min_test_loss:", min_test_loss)


#%%
""" e) We want to extract more information from the data, go back to the scaling 
data level, calculate and add the following two features for each sample. Then
scale the data."""

# Load data again
housing = pd.read_csv('./housing.csv')

# Make and add new features
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["population_per_household"] = housing["population"] / housing["households"]

# re-do previous steps again
housing.fillna(housing.mean(), inplace=True)
housing[housing.select_dtypes(include=['float64']).columns] = scaler.fit_transform(housing[housing.select_dtypes(include=['float64']).columns])
housing["ocean_proximity"].value_counts() # find out what categories exist and how many districts belong to each category
housing = pd.get_dummies(housing)

# Check data
housing.info()
housing

# Split new dataset
X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# evaluate models for best combination of learning rate and regularization coefficient
coef_, train_loss, val_loss, test_loss = train_and_evaluate_model(Best_reg_coef, Best_learning_rate)

# Plot loss curves for training and validation data
plt.plot(coef_)
plt.title("\n".join(wrap(f'Learning rate: {learning_rate}, Regularization coefficient: {reg_coef}, Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}')))
plt.ylabel('Coefficient')
plt.xlabel('Feature')
plt.show()

new_min_test_loss = TestLoss[np.argmin(test_loss)]
print("new_min_test_loss:", new_min_test_loss)








