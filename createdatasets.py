import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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
df = pd.read_csv('Combined_Flights_2022.csv')

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

df = 0

test = pd.concat([test_x, test_y], axis=1)

test.to_csv(r'Testset.csv', index=False, header=True)

test = 0

# SMOTE initialisering
smote = SMOTE(random_state=1)

# Brug SMOTE til at over-sample de underrepræsenterede klasser i træningssættet
train_x, train_y = smote.fit_resample(train_x, train_y)

test = pd.concat([train_x, train_y], axis=1)

test.to_csv(r'Trainset.csv', index=False, header=True)