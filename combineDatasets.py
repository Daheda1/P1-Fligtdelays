import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def process_file(file_name, scaler, first_file):
    print (f"Arbejder på {file_name}")

    df = pd.read_csv(file_name)
    train_y = df.pop("DelayLabel")

    # One-hot encode specific columns
    df = pd.get_dummies(df, columns=['Airline', 'Origin', 'Dest'], dtype=int)

    # Scale specific columns
    columns_to_normalize = ["DepTime", "ArrTime", 'Distance']
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])

    # Adding the label back to the dataframe
    df['DelayLabel'] = train_y

    # Append to the combined file
    with open("traindata.csv", 'a') as f:
        df.to_csv(f, header=first_file, index=False)

def process_and_combine_files(start_index, end_index):
    scaler = StandardScaler()
    first_file = True

    for i in range(start_index, end_index + 1):
        file_name = f"train_resampled_batch_{i}.csv"

        if not os.path.exists(file_name):
            print(f"File not found: {file_name}")
            continue

        # Fit the scaler on the first file
        if first_file:
            df_first = pd.read_csv(file_name)
            columns_to_normalize = ["DepTime", "ArrTime", 'Distance']
            scaler.fit(df_first[columns_to_normalize])

        process_file(file_name, scaler, first_file)
        first_file = False

# Kald funktionen
process_and_combine_files(0, 6)
