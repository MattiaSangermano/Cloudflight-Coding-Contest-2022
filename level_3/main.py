import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys


def create_model():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            units=128,
            input_dim=30,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.L2(0.0001)
            ))
    model.add(
        tf.keras.layers.Dense(
            units=30,
            input_dim=30,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.L2(0.0001)
        ))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


train_path = sys.argv[1]
test_path = sys.argv[2]

df = pd.read_csv(train_path)
test = pd.read_csv(test_path, header=None).rename(columns={0: 'x'})

df = df[(df['y'] == 0) | (df['y'] == 1)]

le = LabelEncoder().fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
df['in_f'] = df['x'].apply(lambda x: le.transform([*x]).tolist())
X = np.array(df['in_f'].tolist())
Y = df['y'].values

test['in_f'] = test['x'].apply(lambda x: le.transform([*x]).tolist())
test_X = np.array(test['in_f'].tolist())

model = create_model()
history = model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=64,
                    verbose=2, shuffle=1)
predictions = (model.predict(test_X) > 0.5) + 0

s = ''
for el in predictions[:, 0].tolist():
    s += str(el) + '\n'
s = s[:-1]

with open(f"level_3_output.txt", 'w') as f:
    f.write(s)
