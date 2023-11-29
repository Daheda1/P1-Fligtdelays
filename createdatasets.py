import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
import warnings
import time
import os
from math import ceil
import gc
warnings.filterwarnings("ignore", category=FutureWarning)


def hot_and_scale(x):
    #One-hot encoder vores koloner
    x = pd.get_dummies(x, columns=['Airline', 'Origin', 'Dest'], dtype=int, sparse=True)
    #skalere vores koloner
    scaler = StandardScaler()
    columns_to_normalize = ["DepTime", "ArrTime", 'Distance']
    x[columns_to_normalize] = scaler.fit_transform(x[columns_to_normalize])
    return x

def combine_and_save(x, y, path):
    temp = pd.concat([ y, x], axis=1)

    temp.to_csv(path, index=False, header=True)

def batch_process_and_save(train_x, train_y, categorical_features_indices, batch_size):
    file_prefix = "train_resampled"
    batch_filename_list = []
    smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
    num_samples = len(train_x)
    num_batches = ceil(num_samples / batch_size)

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, num_samples)

        batch_x = train_x[start_idx:end_idx]
        batch_y = train_y[start_idx:end_idx]

        batch_x_res, batch_y_res = smote_nc.fit_resample(batch_x, batch_y)

        batch_resampled = pd.concat([pd.DataFrame(batch_y_res), pd.DataFrame(batch_x_res)], axis=1)
        batch_filename = f"{file_prefix}_batch_{batch_num}.csv"
        batch_resampled.to_csv(batch_filename, index=False, header=True)

        print(f"Batch {batch_num} processed and saved to {batch_filename}")
        batch_filename_list.append(batch_filename)
    return batch_filename_list

def process_file(file_name, first_file):
    print(f"Samler {file_name} til Trainset.csv")

    df = pd.read_csv(file_name)
    train_y = df.pop("DelayLabel")

    # Apply hot_and_scale function
    df = hot_and_scale(df)

    # Adding the label back to the dataframe
    df['DelayLabel'] = train_y

    # Append to the combined file
    with open("Trainset.csv", 'a') as f:
        df.to_csv(f, header=first_file, index=False)
    os.remove(file_name)

def process_and_combine_files(file_names):
    first_file = True

    for file_name in file_names:
        if not os.path.exists(file_name):
            print(f"File not found: {file_name}")
            continue

        process_file(file_name, first_file)
        first_file = False

#Denne funktion bestemmer hvilket label "DelayLabel" ender på basseret på forsinkelsen 
def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    elif delay <= 120:
        return 'late'
    else:
        return 'very-late'

def get_dataset(nrows):
    #Definere de kolonner vi gerne vil træne på
    relevant_columns = ['Airline', 'Origin', 'Dest', 
                        'DepTime', 'ArrTime', 'Distance', 
                        'DayOfWeek', 'DayofMonth', 'Quarter']
        
    #Henter vores datasæt og laver det til pandas dataframe
    df = pd.read_csv('Combined_Flights_2022.csv', usecols=relevant_columns + ['ArrDelayMinutes'], nrows = nrows)

    #DelayLabel bliver tilføjet og apply bruger funktionen label_delay på hele rækken
    df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)

    df.dropna(inplace=True)

    return df.pop("DelayLabel"), df

def main():
    label, df = get_dataset(5000000)
    print("Datasæt indlæst")

    train_x, test_x, train_y, test_y = train_test_split(df, label, stratify=label, test_size=0.20, random_state=1)
    print("80/20 Split lavet")
    del df
    del label

    test_x = hot_and_scale(test_x)

    combine_and_save(test_x, test_y, "Testset.csv")
    print("Testsæt gemt")
    print("Påbegynder SMOTE")
    categorical_features_indices = [0, 1, 2]
    filenames = batch_process_and_save(train_x, train_y, categorical_features_indices, 500000)
    print("SMOTE Afsluttet")
    process_and_combine_files(filenames)
    print("Træningssæt gemt")

main()