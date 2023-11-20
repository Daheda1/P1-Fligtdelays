import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    precision_score, recall_score, 
    f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#Denne funktion bestemmer hvilket label "DelayLabel" ender på basseret på forsinkelsen 
def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    elif delay <= 120:
        return 'late'
    else:
        return 'very-late'
    
#Henter vores datasæt og laver det til pandas dataframe
df = pd.read_csv('Combined_Flights_2022.csv', nrows = 1000000)

#DelayLabel bliver tilføjet og apply bruger funktionen label_delay på hele rækken
df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)

#Definere de kolonner vi gerne vil træne på
relevant_columns = ['Airline', 'Origin', 'Dest', 
                    'DepTime', 'ArrTime', 'DelayLabel', 
                    'Distance', 'DayOfWeek', 'DayofMonth', 'Quarter']

#Beholder kun de data vi vil træne på
df = df[relevant_columns]

#One-hot encoder vores koloner
df = pd.get_dummies(df, columns=['Airline', 'Origin', 'Dest'], dtype=int)

#skalere vores koloner
scaler = StandardScaler()
columns_to_normalize = ["DepTime", "ArrTime", 'Distance']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# fjerner alle rækker med tomme felter
rows_before = len(df)
df.dropna(inplace=True)
rows_after = len(df)
rows_removed = rows_before - rows_after
print(f"Fjernet {rows_removed} rækker.")

#fjerne DelayLabel fra df og gemmer dem som label
label = df.pop("DelayLabel")

#Laver et 80/20 split på vores data og labels
train_x, test_x, train_y, test_y = train_test_split(df, label, stratify=label, test_size=0.20, random_state=1)

rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
gnb = GaussianNB()
sgd = SGDClassifier()
knc = KNeighborsClassifier()

#svc = SVC()



models = {rfc, dtc, gnb, sgd, knc}

train_x = np.ascontiguousarray(train_x)
test_x = np.ascontiguousarray(test_x)

def evaluate_and_save_metrics(train_x, test_x, train_y, test_y, models):
    results = []
    for model in models:
        # Get the model name
        model_name = model.__class__.__name__
        print(f"køre nu {model_name}")
        # Fit the model
        model.fit(train_x, train_y)
        # Predict on the test set
        predicted_values = model.predict(test_x)
        # Calculate metrics
        acc = accuracy_score(test_y, predicted_values)
        conf_matrix = confusion_matrix(test_y, predicted_values)
        prec = precision_score(test_y, predicted_values, average='weighted', zero_division=0)
        rec = recall_score(test_y, predicted_values, average='weighted', zero_division=0)
        f1 = f1_score(test_y, predicted_values, average='weighted', zero_division=0)

        # Save the metrics to a file
        filename = f'{model_name}.txt'
        with open(filename, 'w') as file:
            file.write(f'Accuracy: {acc}\n')
            file.write(f'Confusion Matrix:\n{conf_matrix}\n')
            file.write(f'Precision (weighted): {prec}\n')
            file.write(f'Recall (weighted): {rec}\n')
            file.write(f'F1 Score (weighted): {f1}\n')
        results.append(filename)
        print(f"Resultater gemt i filen {model_name}.txt")
    return results


print(evaluate_and_save_metrics(train_x, test_x, train_y, test_y, models))