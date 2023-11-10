import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

# Indlæser data
df = pd.read_csv('Combined_Flights_2022.csv')

# Definerer funktionen til at oprette 'DelayLabel'
def label_delay(delay):
    if delay <= 15:
        return 'on-time'
    elif delay <= 45:
        return 'late'
    else:
        return 'very-late'

# Anvender funktionen til at oprette 'DelayLabel'
df['DelayLabel'] = df['DepDelayMinutes'].apply(label_delay)

# Definerer de relevante kolonner
relevant_columns = ['Airline', 'Origin', 'Dest', 'DepTime', 'ArrTime', 'Distance', 'DayOfWeek', 'DayofMonth', 'Quarter']


# Fjerner rækker med NaN-værdier fra df
df = df.dropna(subset=relevant_columns + ['DepDelayMinutes'])

# Kategoriske kolonner der skal encodes
categorical_columns = ['Airline', 'Origin', 'Dest']
# Numeriske kolonner
numerical_columns = [col for col in relevant_columns if col not in categorical_columns]

# Del data op i features og labels
X = df[relevant_columns]
y = df['DelayLabel']

# Encoder kategoriske variabler
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), categorical_columns)],
    remainder='passthrough'
)

# Anvend one-hot encoding til features og labels
X_encoded = ct.fit_transform(X)
y_encoded = pd.get_dummies(y)

# Definerer Keras model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(y_encoded.shape[1], activation='softmax'))  # Antallet af output-neuroner svarer til antallet af klasser
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Input_dim beregnes som længden af det transformeret feature space
input_dim = X_encoded.shape[1]

# Indpakker Keras model som en estimator
keras_classifier = KerasClassifier(build_fn=lambda:create_model(input_dim), epochs=10, batch_size=100, verbose=1)

# Træn modellen
keras_classifier.fit(X_encoded, y_encoded)

# Gemmer hele den underliggende Keras-model til en enkelt HDF5-fil
keras_classifier.model.save('min_model.h5')