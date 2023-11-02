import pandas as pd

def label_delay(delay):
    if delay <= 0:
        return 'on-time'
    elif delay <= 30:
        return 'late'
    else:
        return 'very-late'

# Antag at data er gemt i en fil kaldet 'flight_data.csv'
df = pd.read_csv('Combined_Flights_2022.csv', nrows=300000)

df['DelayLabel'] = df['ArrDelayMinutes'].apply(label_delay)


# Fjern kolonner, der indeholder de samme oplysninger. Vælg de kolonner, der er relevante for modellen.
# Dette er et eksempel og kan variere baseret på dine specifikke behov
relevant_columns = ['Airline', 'Origin', 'Dest', 
                    'DepTime',  'ArrTime', 'DelayLabel', 
                    'Distance', 'DayOfWeek', 'DayofMonth','Quarter', 'DistanceGroup']

df = df[relevant_columns]

# Konverter kategoriske værdier til numeriske værdier ved hjælp af label encoding
# Dette kan også gøres med one-hot encoding, afhængig af den model, du planlægger at bruge
df = pd.get_dummies(df, columns=['Airline', 'Origin', 'Dest'], dtype=int)

# Tjek de første rækker i det rensede dataframe
df.to_csv('cleaned_flight_data.csv', index=False)
print(df.head())
