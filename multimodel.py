import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    precision_score, recall_score, 
    f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

    
#Henter vores datasæt og laver det til pandas dataframe
df = pd.read_csv('Trainset.csv')
print("Datasæt indlæst")


#Laver et 80/20 split på vores data og labels
train_y = df.pop("DelayLabel")

train_x = df

rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
gnb = GaussianNB()
sgd = SGDClassifier()
knc = KNeighborsClassifier()

# SMOTE initialisering
smote = SMOTE(random_state=1)

models = {rfc, dtc, gnb, sgd, knc}

train_x = np.ascontiguousarray(train_x)

def evaluate_and_save_metrics(train_x, train_y, models):
    for model in models:
        # Get the model name
        model_name = model.__class__.__name__
        print(f"køre nu {model_name}")
        # Fit the model
        model.fit(train_x, train_y)
        # Predict on the test set
        joblib.dump(model, f'{model_name}.joblib')  # Save the model


print(evaluate_and_save_metrics(train_x, train_y, models))