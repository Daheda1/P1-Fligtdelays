import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    precision_score, recall_score, 
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#Denne funktion bestemmer hvilket label "DelayLabel" ender på basseret på forsinkelsen 
def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    else:
        return 'late'
#Henter vores datasæt og laver det til pandas dataframe
df = pd.read_csv('Combined_Flights_2022.csv', nrows = 100000)
#DelayLabel bliver tilføjet og apply bruger funktionen label_delay på hele rækken
df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)
#Definere de kolonner vi gerne vil træne på
relevant_columns = ['Airline', 'Origin', 'Dest', 
                    'DepTime', 'ArrTime', 'DelayLabel', 
                    'Distance', 'DayOfWeek', 'DayofMonth', 'Quarter']

#Beholder kun de data vi vil træne på
df = df[relevant_columns]
# fjerner alle rækker med tomme felter
rows_before = len(df)
df.dropna(inplace=True)
rows_after = len(df)
rows_removed = rows_before - rows_after
print(f"Fjernet {rows_removed} rækker.")
#One-hot encoder vores koloner
df = pd.get_dummies(df, columns=['Airline', 'Origin', 'Dest'], dtype=int, sparse=True)
#skalere vores koloner
scaler = StandardScaler()
columns_to_normalize = ["DepTime", "ArrTime", 'Distance']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
#fjerne DelayLabel fra df og gemmer dem som label
label = df.pop("DelayLabel")
#Laver et 80/20 split på vores data og labels
train_x, test_x, train_y, test_y = train_test_split(df, label, stratify=label, test_size=0.20, random_state=1)


def train_on_single_columns(train_x, test_x, train_y, test_y): 
    for column in train_x.columns:
        # Træner modellen på en enkelt kolonne
        model = RandomForestClassifier()
        model.fit(train_x[[column]], train_y)
        
        # Foretager forudsigelser på test-sættet
        predictions = model.predict(test_x[[column]])
        
        # Beregner nøjagtighed for den aktuelle kolonne
        current_accuracy = accuracy_score(test_y, predictions)
        print(f"Nøjagtighed for {column}: {current_accuracy}")

train_on_single_columns(train_x, test_x, train_y, test_y)