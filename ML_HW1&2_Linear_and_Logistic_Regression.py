#=================================================
#   ML_HW1&2_Linear_and_Logistic_Regression
#   Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
#   Using Python 3.9.16
#=================================================

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#%% Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time


#%% Pre-processing the data

# Load the dataset
data = pd.read_csv("./mitbih.csv")


# Check data
data
data.info() # get a quick description of the data
data.describe() # shows a summary of the numerical attributes


# Missing values
print(data.isnull().sum()) #check missing data
data.fillna(data.mean(), inplace=True) # Replace missing values with mean


# Duplicated data
print(data.duplicated().sum()) # check duplicated data | there are some duplicate rows in the dataset since some of the values are True.


# Check data again
data
data.info() # get a quick description of the data
data.describe() # shows a summary of the numerical attributes



#%% Split the data into features and labels
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X = np.array(X)
y = np.array(y)

# Split the data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)



#%% (GRAD) the logistic regression algorithm with the gradient ascent algorithm
""" Q1-a) Implement the logistic regression algorithm with the gradient ascent algorithm. 
By changing the learning rate in the interval [1,0), check the effect of the learning rate 
on the convergence speed. Find the appropriate learning rate and training stopping point
using a validation set that is randomly selected up to 20% of the training data. Then 
plot the accuracy curve on the training and test data. Also find the confusion matrix and 
report the training duration. """

#===============================================
# logistic regression algorithm with the gradient ascent function
class MyLogisticRegression:
    
    # initialising all the parameters
    def __init__(self, learning_rate, max_iterations): 
        # The self in the function just represent the instance of the class. 
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.likelihoods = []
        self.eps = 1e-7 # Define epsilon because log(0) is not defined
    
    # sigmoid used to map predictions to the range of 0 and 1    
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
    
    # cost functions in gradient ascent
    def log_likelihood(self, y_true, y_pred):
        # fix 0/1 values in y_pred so that log is not undefined
        y_pred = np.maximum(np.full(y_pred.shape, self.eps), np.minimum(np.full(y_pred.shape, 1-self.eps), y_pred))
        likelihood = sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        return likelihood
        
    # gradient ascent
    def fit(self, X, y):
        num_features = X.shape[1]
        # Initialize weights with appropriate shape
        self.weights = np.zeros(num_features)
        # Perform gradient ascent
        for i in range(self.max_iterations):
            # define the linear hypothesis(z) first
            z = np.dot(X,self.weights)
            # output probability value by appplying sigmoid on z
            y_pred = self.sigmoid(z)
            
            # calculate the gradient values
            gradient = np.mean((y-y_pred)*X.T, axis=1)
            
            # update the weights
            self.weights = self.weights +self.learning_rate*gradient
            # calculating log likelihood
            likelihood = self.log_likelihood(y,y_pred)
            self.likelihoods.append(likelihood)
    
    def predict_proba(self,X):
        if self.weights is None:
            raise Exception("Fit the model before prediction")
        z = np.dot(X,self.weights)
        probabilities = self.sigmoid(z)
        return probabilities
    
    # Thresholding probability to predict binary values
    def predict(self, X, threshold=0.5):
        binary_predictions = np.array(list(map(lambda x: 1 if x>threshold else 0, self.predict_proba(X))))
        return binary_predictions

#===============================================
def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
#===============================================


#===============================================
learning_rates = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
max_iterations = 100
#===============================================


#===============================================
# Find the best learning rate by 20% of the training data (validation data)
accuracies_val_lr = []
confusion_matrices_val_lr = []
Time_val_GRAD = []

# Validation
for learning_rate in learning_rates:
    # start time caculating
    start = time.time()
    # function
    model_GRAD = MyLogisticRegression(learning_rate, max_iterations)
    model_GRAD.fit(X_train,y_train)
    # calculate y_pred
    y_val_pred_lr = model_GRAD.predict(X_val)
    # calculate accuracy
    accuracies_val_lr.append(accuracy(y_val, y_val_pred_lr))
    # calculate confusion matrix
    confusion_matrices_val_lr.append(confusion_matrix(y_val, y_val_pred_lr))
    # end time caculating
    end = time.time()
    t = (end - start)
    Time_val_GRAD.append(t)

# Find the appropriate learning rate and training stopping point (Validation)
best_learning_rate_lr = learning_rates[np.argmax(accuracies_val_lr)]
accuracy_best_learning_rate_lr = np.max(accuracies_val_lr)
print("Best learning rate (GRAD):", best_learning_rate_lr)
print("Accuracy of GRAD (best learning rate):", accuracy_best_learning_rate_lr)


# Plot time curves (Validation)
plt.plot(learning_rates, Time_val_GRAD, label='Validation')
plt.xlabel('Learning Rate')
plt.ylabel('Time')
plt.title('Time vs. learning rate - GRAD')
plt.legend()
plt.show()

# Plot accuracy curves (Validation)
plt.plot(learning_rates, accuracies_val_lr, label='Validation')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. learning rate - GRAD')
plt.legend()
plt.show()

# Calculate confusion matrix (Validation)
cm_val_GRAD = confusion_matrices_val_lr[np.argmax(accuracies_val_lr)]
sns.heatmap(cm_val_GRAD, annot=True, cmap='jet')
plt.title('Confusion matrix of GRAD (Validation)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#===============================================


#===============================================
accuracies_train_grad = []
accuracies_test_grad = []
confusion_matrices_train_grad = []
confusion_matrices_test_grad = []
Time_grad = []

epoch = []

for i in range(max_iterations):
    epoch.append(i)
    # function
    start = time.time() # start time caculating
    model_GRAD = MyLogisticRegression(best_learning_rate_lr, max_iterations)
    model_GRAD.fit(X_train,y_train)
    end = time.time() # end time caculating
    t = (end - start)
    Time_grad.append(t)
    
    # Train
    y_train_pred_grad = model_GRAD.predict(X_train) # calculate y_pred
    accuracies_train_grad.append(accuracy(y_train, y_train_pred_grad)) # calculate accuracy
    confusion_matrices_train_grad.append(confusion_matrix(y_train, y_train_pred_grad)) # calculate confusion matrix
    
    # Test    
    y_test_pred_grad = model_GRAD.predict(X_test) # calculate y_pred
    accuracies_test_grad.append(accuracy(y_test, y_test_pred_grad)) # calculate accuracy
    confusion_matrices_test_grad.append(confusion_matrix(y_test, y_test_pred_grad)) # calculate confusion matrix

# Plot Time
plt.plot(epoch, Time_grad)
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.title('Time vs. Epoch - GRAD')
plt.legend()
plt.show()

# Plot accuracy curves
plt.plot(epoch, accuracies_train_grad, label='Train')
plt.plot(epoch, accuracies_test_grad, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch - GRAD')
plt.legend(['Train', 'Test'])
plt.show()

# Calculate confusion matrix (Train)
cm_train_GRAD = confusion_matrices_train_grad[np.argmax(accuracies_train_grad)]
sns.heatmap(cm_train_GRAD, annot=True, cmap='jet')
plt.title('Confusion matrix of GRAD (Train)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Calculate confusion matrix (Test)
cm_test_GRAD = confusion_matrices_test_grad[np.argmax(accuracies_test_grad)]
sns.heatmap(cm_test_GRAD, annot=True, cmap='jet')
plt.title('Confusion matrix of GRAD (Test)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#===============================================



#%% (GNB) GNB algorithm
""" Q1-b) Implement a GNB algorithm and compare the results with 1(a). To avoid
zero probability, consider the prior distribution as a Gaussian distribution 
(hint: Be sure to use the logarithm). Also find the confusion matrix and 
report the training duration."""
  
gnb = GaussianNB()

accuracies_train_gnb = []
accuracies_test_gnb = []
confusion_matrices_train_gnb = []
confusion_matrices_test_gnb = []
Time_gnb = []

for i in range(max_iterations):
    # function
    start = time.time() # start time caculating
    model_gnb = GaussianNB()
    model_gnb.fit(X_train,y_train)
    end = time.time() # end time caculating
    t = (end - start)
    Time_gnb.append(t)
    
    # Train
    y_train_pred_gnb = model_gnb.predict(X_train)
    accuracies_train_gnb.append(accuracy(y_train, y_train_pred_gnb))
    confusion_matrices_train_gnb.append(confusion_matrix(y_train, y_train_pred_gnb))
    
    # Test    
    y_test_pred_gnb = model_gnb.predict(X_test)
    accuracies_test_gnb.append(accuracy(y_test, y_test_pred_gnb))
    confusion_matrices_test_gnb.append(confusion_matrix(y_test, y_test_pred_gnb))
    
# Plot Time
plt.plot(epoch, Time_gnb)
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.title('Time vs. Epoch - GRAD')
plt.legend()
plt.show()

# Plot accuracy curves
plt.plot(epoch, accuracies_train_gnb, label='Train')
plt.plot(epoch, accuracies_test_gnb, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch - GNB')
plt.legend(['Train', 'Test'])
plt.show()

# Calculate confusion matrix (Train)
cm_train_gnb = confusion_matrices_train_gnb[np.argmax(accuracies_train_gnb)]
sns.heatmap(cm_train_gnb, annot=True, cmap='jet')
plt.title('Confusion matrix of GNB (Train)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Calculate confusion matrix (Test)
cm_test_gnb = confusion_matrices_test_gnb[np.argmax(accuracies_test_gnb)]
sns.heatmap(cm_test_gnb, annot=True, cmap='jet')
plt.title('Confusion matrix of GNB (Test)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#===============================================



#%% (LogReg) logistic regression algorithm
""" Q1-c) Implement the logistic regression algorithm with the regularization 
form by choosing the appropriate parameter, and compare it with result of 
question 1(a) and 1(b). Also find the confusion matrix and report the training 
duration. """

accuracies_train_logreg = []
accuracies_test_logreg = []
confusion_matrices_train_logreg = []
confusion_matrices_test_logreg = []
Time_logreg = []

for i in range(max_iterations):
    # function
    start = time.time() # start time caculating
    model_logreg = LogisticRegression(solver='lbfgs', max_iter=max_iterations, C=1/best_learning_rate_lr)
    model_logreg.fit(X_train,y_train)
    end = time.time() # end time caculating
    t = (end - start)
    Time_logreg.append(t)
    
    # Train
    y_train_pred_logreg = model_logreg.predict(X_train)
    accuracies_train_logreg.append(accuracy(y_train, y_train_pred_logreg))
    confusion_matrices_train_logreg.append(confusion_matrix(y_train, y_train_pred_logreg))
    
    # Test    
    y_test_pred_logreg = model_logreg.predict(X_test)
    accuracies_test_logreg.append(accuracy(y_test, y_test_pred_logreg))
    confusion_matrices_test_logreg.append(confusion_matrix(y_test, y_test_pred_logreg))

# Plot Time
plt.plot(epoch, Time_logreg)
plt.xlabel('Epoch')
plt.ylabel('Time')
plt.title('Time vs. Epoch - GRAD')
plt.legend()
plt.show()

# Plot accuracy curves
plt.plot(epoch, accuracies_train_logreg, label='Train')
plt.plot(epoch, accuracies_test_logreg, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch - LogReg')
plt.legend(['Train', 'Test'])
plt.show()

# Calculate confusion matrix (Train)
cm_train_logreg = confusion_matrices_train_logreg[np.argmax(accuracies_train_logreg)]
sns.heatmap(cm_train_logreg, annot=True, cmap='jet')
plt.title('Confusion matrix of LogReg (Train)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Calculate confusion matrix (Test)
cm_test_logreg = confusion_matrices_test_logreg[np.argmax(accuracies_test_logreg)]
sns.heatmap(cm_test_logreg, annot=True, cmap='jet')
plt.title('Confusion matrix of LogReg (Test)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#===============================================

################################################
# Plot accuracy curves
plt.plot(epoch, accuracies_train_grad, label='Train_GRAD')
plt.plot(epoch, accuracies_test_grad, label='Train_GRAD')
plt.plot(epoch, accuracies_train_gnb, label='Train_GNB')
plt.plot(epoch, accuracies_test_gnb, label='Test_GNB')
plt.plot(epoch, accuracies_train_logreg, label='Train_LogReg')
plt.plot(epoch, accuracies_test_logreg, label='Test_LogReg')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend(['Train_GRAD','Train_GRAD','Train_GNB','Test_GNB','Train_LogReg','Test_LogReg'])
plt.show()
################################################


#%% plot the training and test curve for logistic regression and GNB
""" Q1-d) For logistic regression (the best result) and GNB, plot the training and test 
curve based on different training samples (classification accuracy rate based on the 
number of training samples). For this purpose, start the amount training data from a 
minimum of a quarter of the number of each class, and take the step of increasing about 
one eighth of the amount of data of that class."""



#%%
""" Q2-a) For training of the logistic regression of question 1(a) use 5-fold-cross 
validation. Then test it on the entire test data and finally report the average 
classification accuracy on the training and test data. Repeat this act of randomly 
dividing the data for training 5 times and determine the accuracy values each time, 
and finally get the average accuracy. Compare all the results with result of question 
1(a) and 1(b)."""

# training 5 times and determine the accuracy values each time
random_test = 5
mean_scores_train_cv_grad = []
mean_scores_test_cv_grad = []
mean_scores_test_CV_grad = []

for j in range(random_test):
    
    # Randomizing
    data_copy = data
    data_copy = data_copy.reindex(np.random.permutation(data_copy.index))
    # Reset index
    data_copy = data_copy.reset_index(drop=True)
    # split train and test
    train, test = train_test_split(data_copy, train_size=0.8, random_state=42)
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]
    
    """ On Original Train Data """ 
    # split train data to train_cv and test_cv
    
    # folds | number of cross-validation (cv)
    cv = 5
    # split train to "cv" parts
    data_splited = np.array_split(train, cv) 
    fold = [0] * cv
    train_cv_grad = [0] * cv
    test_cv_grad = [0] * cv
    
    scores_train_cv_grad = []
    scores_test_cv_grad = []
    scores_test_CV_grad = []
    
    for i in range(cv):
        fold[i] = data_splited[i]
    for i in range(cv):
        train_cv_grad[i] = pd.concat(fold[:i] + fold[i+1:]) # Concatenate pandas objects along a particular axis
        test_cv_grad[i] = fold[i]
        X_train_cv_grad = train_cv_grad[i].iloc[:,:-1]
        y_train_cv_grad = train_cv_grad[i].iloc[:,-1] 
        X_test_cv_grad = test_cv_grad[i].iloc[:,:-1]
        y_test_cv_grad = test_cv_grad[i].iloc[:,-1]
                
        model_cv_GRAD = MyLogisticRegression(best_learning_rate_lr, max_iterations)
        model_cv_GRAD.fit(X_train_cv_grad, y_train_cv_grad)
        
        # Train_cv
        y_train_pred_cv_grad = model_GRAD.predict(X_train_cv_grad)
        score_train_cv_grad = accuracy(y_train_cv_grad, y_train_pred_cv_grad)
        scores_train_cv_grad.append(score_train_cv_grad)
        
        # Test_cv
        y_test_pred_cv_grad = model_GRAD.predict(X_test_cv_grad)
        score_test_grad = accuracy(y_test_cv_grad, y_test_pred_cv_grad)
        scores_test_cv_grad.append(score_test_grad)

        # # Test
        # """ On Original Test Data """
        # y_test_pred_CV_grad = model_GRAD.predict(X_test_cv_grad)
        # score_testCV_grad = accuracy(y_test_cv_grad, y_test_pred_CV_grad)
        # scores_test_CV_grad.append(score_testCV_grad)        
        
        
    mean_scores_train_cv_grad.append(np.mean(scores_train_cv_grad))
    mean_scores_test_cv_grad.append(np.mean(scores_test_cv_grad))
    # mean_scores_test_CV_grad.append(np.mean(scores_test_CV_grad))

    
print("Mean accuracy of GRAD-cv (Train):", mean_scores_train_cv_grad)
print("Mean accuracy of GRAD-cv (Test from folds):", mean_scores_test_cv_grad)
# print("Mean accuracy of GRAD-cv (Test -original data):", mean_scores_test_CV_grad)



#%%
""" Q2-b) Do the previous step this time for the regularized form of logistic
regression. Compare all the results with results of question 1(a) and 1(b) and
2(a)."""


# training 5 times and determine the accuracy values each time
random_test = 5
mean_scores_train_cv_logreg = []
mean_scores_test_cv_logreg = []
mean_scores_test_CV_logreg = []

for j in range(random_test):
    
    # Randomizing
    data_copy = data
    data_copy = data_copy.reindex(np.random.permutation(data_copy.index))
    # Reset index
    data_copy = data_copy.reset_index(drop=True)
    # split train and test
    train, test = train_test_split(data_copy, train_size=0.8, random_state=42)
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]
    
    """ On Original Train Data """ 
    # split train data to train_cv and test_cv
    
    # folds | number of cross-validation (cv)
    cv = 5
    # split train to "cv" parts
    data_splited = np.array_split(train, cv) 
    fold = [0] * cv
    train_cv_logreg = [0] * cv
    test_cv_logreg = [0] * cv
    
    scores_train_cv_logreg = []
    scores_test_cv_logreg = []
    scores_test_CV_logreg = []
    
    for i in range(cv):
        fold[i] = data_splited[i]
    for i in range(cv):
        train_cv_logreg[i] = pd.concat(fold[:i] + fold[i+1:]) # Concatenate pandas objects along a particular axis
        test_cv_logreg[i] = fold[i]
        X_train_cv_logreg = train_cv_logreg[i].iloc[:,:-1]
        y_train_cv_logreg = train_cv_logreg[i].iloc[:,-1] 
        X_test_cv_logreg = test_cv_logreg[i].iloc[:,:-1]
        y_test_cv_logreg = test_cv_logreg[i].iloc[:,-1]
                
        model_cv_logreg = LogisticRegression(solver='lbfgs', max_iter=max_iterations, C=1/best_learning_rate_lr)
        model_cv_logreg.fit(X_train_cv_logreg, y_train_cv_logreg)
        
        # Train_cv
        y_train_pred_cv_logreg = model_logreg.predict(X_train_cv_logreg)
        score_train_cv_logreg = accuracy(y_train_cv_logreg, y_train_pred_cv_logreg)
        scores_train_cv_logreg.append(score_train_cv_logreg)
        
        # Test_cv
        y_test_pred_cv_logreg = model_logreg.predict(X_test_cv_logreg)
        score_test_logreg = accuracy(y_test_cv_logreg, y_test_pred_cv_logreg)
        scores_test_cv_logreg.append(score_test_logreg)

        # # Test
        # """ On Original Test Data """
        # y_test_pred_CV_logreg = model_logreg.predict(X_test_cv_logreg)
        # score_testCV_logreg = accuracy(y_test, y_test_pred_CV_logreg)
        # scores_test_CV_logreg.append(score_testCV_logreg)        
        
        
    mean_scores_train_cv_logreg.append(np.mean(scores_train_cv_logreg))
    mean_scores_test_cv_logreg.append(np.mean(scores_test_cv_logreg))
    # mean_scores_test_CV_logreg.append(np.mean(scores_test_CV_logreg))
    
print("Mean accuracy of LogReg-cv (Train):", mean_scores_train_cv_logreg)
print("Mean accuracy of LogReg-cv (Test from folds):", mean_scores_test_cv_logreg)
# print("Mean accuracy of LogReg-cv (Test -original data):", mean_scores_test_CV_logreg)

