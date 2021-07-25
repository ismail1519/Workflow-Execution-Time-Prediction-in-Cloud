import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/ismai/PycharmProjects/pythonProject/data.csv')
print(df,sep=',')
df.head()
print("With Normalization")
dataset = df.loc[:,'bandwidth'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'peakmemory'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'ioread'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'iowrite'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'task'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'memory'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'core'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'avg cpu'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'max cpu'].values
dataset = np.reshape(dataset,(-1,1))
dataset = df.loc[:,'min cpu'].values
dataset = np.reshape(dataset,(-1,1))
print(dataset)
dataset.shape



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset)


from numpy import array
def split_sequence(sequence, n_steps):
  x, y = list(), list()
  for i in range(len(sequence)):
    end_ix = i + n_steps
    if end_ix > len(sequence)-1:
      break
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    x.append(seq_x)
    y.append(seq_y)
  return array(x), array(y)

series = array(dataset)
print(series.shape)

x, y = split_sequence(series, 365)
print(x.shape, y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#reshape
series1 = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
series2 = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(series1.shape)
print(series2.shape)

#BLSTM implementation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

K.clear_session()
model = Sequential()
model.add(Bidirectional(LSTM(13), input_shape=(365, 1)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(series1, y_train, epochs=100, batch_size= 200, validation_data=(series2, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
model.summary()


#prediction
train_predict = model.predict(series1)
test_predict = model.predict(series2)
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

#error calculation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print('Train Mean Absolute Error:', mean_absolute_error(y_train[:,0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(y_train[:,0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(y_test[:,0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(y_test[:,0], test_predict[:,0])))

#plot model loss
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();


#actual vs prediction plotting
import seaborn as sns
aa=[x for x in range(200)]
plt.figure(figsize=(12,6))
plt.plot(aa, y_test[:,0][:200], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Execution Time', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

print(test_predict.shape)
print(y_test.shape)

a = plt.axes(aspect='equal')
plt.scatter(y_test, test_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 220]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


import scipy.stats as measures
per_coef = measures.pearsonr(y_test[:,0], test_predict[:,0])
print(per_coef)
per_coef1 = measures.pearsonr(y_train[:,0], train_predict[:,0])
print(per_coef1)


