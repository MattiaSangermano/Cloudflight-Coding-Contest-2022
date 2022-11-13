import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[30]))
    model.add(tf.keras.layers.LSTM(units=256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=128))
    model.add(
        tf.keras.layers.Dense(
            units=64,
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
    model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


train_path = sys.argv[1]
test_path = sys.argv[2]
output_filename = sys.argv[3]

df = pd.read_csv(train_path)
test = pd.read_csv(test_path, header=None).rename(columns={0: 'x'})

le = LabelEncoder().fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
df['in_f'] = df['x'].apply(lambda x: le.transform([*x]).tolist())

train, val = train_test_split(
            df,
            train_size=0.9,
            stratify=df['y']
        )

oversampler = RandomOverSampler(sampling_strategy='all')
train, _ = oversampler.fit_resample(train, train['y'])

train_x = np.array(train['in_f'].tolist())
train_y = train['y'].values

val_x = np.array(val['in_f'].tolist())
val_y = val['y'].values


test['in_f'] = test['x'].apply(lambda x: le.transform([*x]).tolist())
test_X = np.array(test['in_f'].tolist())

class_weight = compute_class_weight(
    'balanced',
    classes=np.unique(train_y),
    y=train_y
)
class_weight = {i:class_weight[i] for i in range(len(class_weight))}

model = create_model()

history = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                    epochs=100, batch_size=64, verbose=2, shuffle=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            mode="max",
                            monitor="val_accuracy",
                            patience=10,
                            restore_best_weights=True,
                            verbose=True)
                            ])

predictions = np.argmax(model.predict(val_x), axis=1)

with open('report.txt', 'w') as report_file:
    report_file.writelines(classification_report(val_y, predictions))

predictions = np.argmax(model.predict(test_X), axis=1)

s = ''
for el in predictions.tolist():
    s += str(el) + '\n'
s = s[:-1]

with open(output_filename, 'w') as f:
    f.write(s)
