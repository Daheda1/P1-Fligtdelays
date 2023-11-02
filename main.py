import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    precision_score, recall_score, 
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    elif delay <= 45:
        return 'late'
    else:
        return 'very-late'

# Antager at data findes i 'Combined_Flights_2022.csv'
df = pd.read_csv('Combined_Flights_2022.csv', nrows=1000000)

df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)

relevant_columns = ['Airline', 'Origin', 'Dest', 
                    'DepTime', 'ArrTime', 'DelayLabel', 
                    'Distance', 'DayOfWeek', 'DayofMonth', 'Quarter']

df = df[relevant_columns]
df = pd.get_dummies(df, columns=['Airline', 'Origin', 'Dest'], dtype=int)
columns_to_normalize = ["DepTime", "ArrTime", 'Distance']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Tæl antallet af rækker før dropna()
rows_before = len(df)
df.dropna(inplace=True)
rows_after = len(df)
rows_removed = rows_before - rows_after
print(f"Fjernet {rows_removed} rækker.")

label = df.pop("DelayLabel")
train_x, test_x, train_y, test_y = train_test_split(df, label, stratify=label, test_size=0.20, random_state=1)

gnb = RandomForestClassifier(n_estimators=100, verbose=1)
model = gnb.fit(train_x, train_y)
predicted_values = gnb.predict(test_x)

acc = accuracy_score(test_y, predicted_values)
conf_matrix = confusion_matrix(test_y, predicted_values)
prec = precision_score(test_y, predicted_values, average='weighted')
rec = recall_score(test_y, predicted_values, average='weighted')

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")

# Print Confusion Matrix with labels
classes = model.classes_
print("Confusion Matrix:")
print(f"Labels: {classes}")
print(conf_matrix)

# For demonstration, let's just consider the 'very-late' label for ROC
if 'very-late' in model.classes_:
    pos_label_idx = list(model.classes_).index('very-late')
    fpr, tpr, _ = roc_curve(test_y, model.predict_proba(test_x)[:, pos_label_idx], pos_label='very-late')
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (area = {roc_auc})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
