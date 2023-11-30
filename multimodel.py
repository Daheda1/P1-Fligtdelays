import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Henter vores datasæt og laver det til en pandas dataframe
df = pd.read_csv('Trainset.csv')
print("Datasæt indlæst")

# Opdeling af data og labels
train_y = df.pop("DelayLabel")
train_x = df

# Initialisering af modeller
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
gnb = GaussianNB()
sgd = SGDClassifier()
knc = KNeighborsClassifier()

# SMOTE initialisering
smote = SMOTE(random_state=1)

# Liste af modeller til krydsvalidering og gemning
models = [rfc, dtc, gnb, sgd, knc]

# Konverter train_x til et contiguous array for at undgå potentielle problemer
train_x = np.ascontiguousarray(train_x)

def evaluate_and_save_models(train_x, train_y, models):
    for model in models:
        # Modelnavn
        model_name = model.__class__.__name__
        print(f"Kører nu krydsvalidering for {model_name}")

        # Anvend SMOTE
        X_res, y_res = smote.fit_resample(train_x, train_y)

        # Krydsvalidering
        scores = cross_val_score(model, X_res, y_res, cv=5)

        # Træne model på hele datasættet
        model.fit(X_res, y_res)

        # Gemme den trænede model
        joblib.dump(model, f'{model_name}.joblib')

        # Gemme krydsvalideringsscoren i en tekstfil
        with open(f'{model_name}_cross_val_scores.txt', 'w') as file:
            file.write(f"Krydsvalideringsscores for {model_name}:\n")
            file.write("\n".join([str(score) for score in scores]))

# Kør evalueringen og gem modellerne og krydsvalideringsscorerne
evaluate_and_save_models(train_x, train_y, models)
