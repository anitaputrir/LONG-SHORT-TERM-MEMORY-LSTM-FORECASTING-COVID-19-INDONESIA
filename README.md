# LONG-SHORT-TERM-MEMORY-LSTM-FORECASTING-COVID-19-INDONESIA
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:56:12 2021

@author: USER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import seaborn as sns
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")

file_ = 'dataset_ga.csv'
df = pd.read_csv(file_)
print(df.head(10))
print(df.info())
print(df.sort_values('date', inplace=True, ignore_index=True))
print(df.head())
df1 = df[-24*365:].reset_index(drop=True)
print(df1.head())
plt.figure(figsize=(15,8))
sns.lineplot(data=df1, x='date', y='total_deaths')
print(df1.describe())

# split data
train_size = int(len(df1) * 0.7) # Menentukan banyaknya data train yaitu sebesar 70% data
train = df1[:train_size]
test =df1[train_size:].reset_index(drop=True)

#Feature Scalling Menggunakan MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train[['total_deaths']])

train['scaled'] = scaler.transform(train[['total_deaths']])
test['scaled'] = scaler.transform(test[['total_deaths']])
#melihat data yang sudah di scalling
print(train.head())

#Membuat fungsi sliding window 
def sliding_window(data, window_size):
    sub_seq, next_values = [], []
    for i in range(len(data)-window_size):
        sub_seq.append(data[i:i+window_size])
        next_values.append(data[i+window_size])
    X = np.stack(sub_seq)
    y = np.array(next_values)
    return X,y

window_size = 24

X_train, y_train = sliding_window(train[['scaled']].values, window_size)
X_test, y_test = sliding_window(test[['scaled']].values, window_size)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Membuat Fungsi Model Forecasting Menggunakan LSTM
def create_model(LSTM_unit=64, dropout=0.2):
    # create model
    model = Sequential()
    model.add(LSTM(units=LSTM_unit, input_shape=(window_size, 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

#Membuat Model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
# Early Stopping
es = EarlyStopping(monitor = 'val_loss', mode = "min", patience = 5, verbose = 0)

# create model
model = KerasRegressor(build_fn=create_model, epochs=50, validation_split=0.1, batch_size=32, callbacks=[es], verbose=1)

# define the grid search parameters
LSTM_unit = [16,32,64,128]
dropout=[0.1,0.2]
param_grid = dict(LSTM_unit=LSTM_unit, dropout=dropout)

#Membuat model GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

#Training model GridSearchCV
grid_result = grid.fit(X_train, y_train)

 #summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# Mengambil model terbaik
best_model = grid_result.best_estimator_.model

#Kemudian coba kita lihat grafik loss function MSE dan metric MAE terhadap epoch untuk melihat performa model terbaik kita dengan cara sebagai berikut
history = best_model.history
# grafik loss function MSE
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('loss function MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
# grafik metric MAE

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('metric MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

#EVALUASI MODEL
# Prediksi data train
predict_train = scaler.inverse_transform(best_model.predict(X_train))
true_train = scaler.inverse_transform(y_train)

# Prediksi data test
predict_test = scaler.inverse_transform(best_model.predict(X_test))
true_test = scaler.inverse_transform(y_test)

# Mean Absolute Error (MAE) data train
mae_train = np.mean(np.abs(true_train-predict_train))
print('MAE data train sebesar:', mae_train)

# Mean Absolute Error (MAE) test data
mae_test = np.mean(np.abs(true_test-predict_test))
print('MAE data test sebesar:', mae_test)

abs_error_train = np.abs(true_train-predict_train)
sns.boxplot(y=abs_error_train)

abs_error_test = np.abs(true_test-predict_test)
sns.boxplot(y=abs_error_test)

#Melihat range data
print('range data train', true_train.max()-true_train.min())
print('range data test', true_test.max()-true_test.min())

#Plot prediksi data train
train['predict'] = np.nan
train['predict'][-len(predict_train):] = predict_train[:,0]

plt.figure(figsize=(15,8))
sns.lineplot(data=train, x='date', y='total_deaths', label = 'train')
sns.lineplot(data=train, x='date', y='predict', label = 'predict')

#Plot prediksi data test
test['predict'] = np.nan
test['predict'][-len(predict_test):] = predict_test[:,0]

plt.figure(figsize=(15,8))
sns.lineplot(data=test, x='date', y='total_deaths', label = 'test')
sns.lineplot(data=test, x='date', y='predict', label = 'predict')

#Plot prediksi sebulan terakhir
plt.figure(figsize=(15,8))
sns.lineplot(data=test[-24*30:], x='date', y='total_deaths', label = 'test')
sns.lineplot(data=test[-24*30:], x='date', y='predict', label = 'predict')

# forecasting data selanjutnya
y_test = scaler.transform(test[['total_deaths']])
n_future = 24*7
future = [[y_test[-1,0]]]
X_new = y_test[-window_size:,0].tolist()

for i in range(n_future):
    y_future = best_model.predict(np.array([X_new]).reshape(1,window_size,1))
    future.append([y_future[0,0]])
    X_new = X_new[1:]
    X_new.append(y_future[0,0])

future = scaler.inverse_transform(np.array(future))
date_future = pd.date_range(start=test['date'].values[-1], periods=n_future+1, freq='H')


             
# Plot Data sebulan terakhir dan seminggu ke depan
plt.figure(figsize=(15,8))
sns.lineplot(data=test[-24*30:], x='date', y='total_deaths', label = 'test')
sns.lineplot(data=test[-24*30:], x='date', y='predict', label = 'predict')
sns.lineplot(x=date_future, y=future[:,0], label = 'future')
plt.ylabel('Total_deaths');




