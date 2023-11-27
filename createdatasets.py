import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings
import os
from math import ceil

import gc
warnings.filterwarnings("ignore", category=FutureWarning)


#Denne funktion bestemmer hvilket label "DelayLabel" ender på basseret på forsinkelsen 
def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    elif delay <= 120:
        return 'late'
    else:
        return 'very-late'

#Definere de kolonner vi gerne vil træne på
relevant_columns = ['Airline', 'Origin', 'Dest', 
                    'DepTime', 'ArrTime', 'DelayLabel', 
                    'Distance', 'DayOfWeek', 'DayofMonth', 'Quarter']

dtype_dict = {'Airline': 'category', 'Origin': 'category', 'Dest': 'category',
              'DepTime': 'float32', 'ArrTime': 'float32', 'Distance': 'float32',
              'DayOfWeek': 'int8', 'DayofMonth': 'int8', 'Quarter': 'int8'}
    
#Henter vores datasæt og laver det til pandas dataframe
df = pd.read_csv('Combined_Flights_2022.csv', usecols=relevant_columns + ['ArrDelayMinutes'], dtype=dtype_dict)

#DelayLabel bliver tilføjet og apply bruger funktionen label_delay på hele rækken
df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)

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

# Clear the memory of the original dataframe
del df
gc.collect()

test = pd.concat([test_x, test_y], axis=1)

test.to_csv(r'Testset.csv', index=False, header=True)

del test
gc.collect()



def save_parts_to_temp_files(X, y, num_parts=5):
    part_size = ceil(len(X) / num_parts)
    temp_files = []

    for part in range(num_parts):
        start_idx = part * part_size
        end_idx = min((part + 1) * part_size, len(X))

        part_data = pd.concat([X.iloc[start_idx:end_idx], y.iloc[start_idx:end_idx]], axis=1)
        temp_file_name = f'temp_part_{part}.csv'
        part_data.to_csv(temp_file_name, index=False)
        temp_files.append(temp_file_name)

    return temp_files

def smote_and_merge_to_single_csv(temp_files, final_csv='Trainset.csv', random_state=1):
    smote = SMOTE(random_state=random_state)
    first_file = True

    for temp_file in temp_files:
        part_data = pd.read_csv(temp_file)
        X_part, y_part = part_data.iloc[:, :-1], part_data.iloc[:, -1]

        X_resampled, y_resampled = smote.fit_resample(X_part, y_part)
        resampled_part = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)

        if first_file:
            resampled_part.to_csv(final_csv, index=False, mode='w', header=True)
            first_file = False
        else:
            resampled_part.to_csv(final_csv, index=False, mode='a', header=False)

        del part_data, X_part, y_part, X_resampled, y_resampled, resampled_part  # Frigør hukommelse

    # Slet midlertidige filer
    for temp_file in temp_files:
        os.remove(temp_file)


temp_files = save_parts_to_temp_files(train_x, train_y, num_parts=5)
smote_and_merge_to_single_csv(temp_files)