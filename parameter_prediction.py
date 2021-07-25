#import libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

#read dataset
tt1 = pd.read_csv('C:/Users/ismai/PycharmProjects/pythonProject/data.csv')

tt1.head()
#dataset preprocessing
tt1['timestamp'] =  pd.to_datetime(tt1['timestamp'])
tt1 = tt1.set_index('timestamp')
tt1.head()
tt = tt1
dataset = tt.values
dataset = dataset.astype('float32')
len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

def create_training_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :11]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)
look_back = 5
trainX, trainY = create_training_dataset(train, look_back=look_back)
testX, testY = create_training_dataset(test, look_back=look_back)

#creating GRU model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(tf.keras.layers.Dense(11))
adamOpt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adamOpt, metrics=['mae'])
history = model.fit(trainX, trainY, validation_split=0.25,epochs=20, batch_size=64, verbose=2)

#prediction
print("Predicting")
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainY = scaler.inverse_transform(trainY)
trainPredict = scaler.inverse_transform(trainPredict)
testY = scaler.inverse_transform(testY)
testPredict = scaler.inverse_transform(testPredict)


#error calculation
print("Evaluating Model")
trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:]))
print('Test Score: %.2f RMSE' % (testScore))

print("Evaluation metrics: MAE(Mean Absolute Error)")

from sklearn.metrics import mean_absolute_error
trainScore = (mean_absolute_error(trainY[:], trainPredict[:]))
print('Train Score: %f MAE' % (trainScore))
testScore = math.sqrt(mean_absolute_error(testY[:], testPredict[:]))
print('Test Score: %f MAE' % (testScore))

trainScore3 = np.corrcoef(trainPredict, trainY)[0,1]
print('Train Correlation: %f COR' % (trainScore3))
testScore3 = np.corrcoef(testPredict, testY)[0,1]
print('Test Correlation: %f COR' % (testScore3))

index=tt.index
TestY= pd.DataFrame(testY,columns=['min_cpu', 'max_cpu', 'avg_cpu', 'task_cpu', 'peakmemory','bandwidth', 'ioread', 'iowrite', 'core', 'memory', 'os'])
PredY=pd.DataFrame(testPredict,columns=['min_cpu', 'max_cpu', 'avg_cpu', 'task_cpu', 'peakmemory', 'bandwidth', 'ioread', 'iowrite', 'core', 'memory', 'os'])


#plotting results
x=index[-1722:]

plt.plot(x,TestY.min_cpu,'.',label='Test min cpu',color='red')
plt.plot(x,PredY.min_cpu,'--',label='Predicted min cpu',color='black')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('CPU Utilization')
plt.show()

plt.plot(x,TestY.max_cpu,'.',label='Test max cpu',color='magenta')
plt.plot(x,PredY.max_cpu,'--',label='Predicted max cpu',color='navy')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('CPU Utilization')
plt.show()

plt.plot(x,TestY.avg_cpu,'.',label='Test avg avg cpu',color='orange')
plt.plot(x,PredY.avg_cpu,'--',label='Predicted avg avg cpu',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('CPU Utilization')
plt.show()

plt.plot(x,TestY.task_cpu,'.',label='Test avg task_cpu',color='orange')
plt.plot(x,PredY.task_cpu,'--',label='Predicted avg task_cpu',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('No.of Tasks')
plt.show()

plt.plot(x,TestY.peakmemory,'.',label='Test avg peakmemory',color='orange')
plt.plot(x,PredY.peakmemory,'--',label='Predicted avg peakmemory',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Memory Utilization')
plt.show()

plt.plot(x,TestY.memory,'.',label='Test avg memory',color='orange')
plt.plot(x,PredY.memory,'--',label='Predicted avg memory',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Memory Utilization')
plt.show()

plt.plot(x,TestY.bandwidth,'.',label='Test avg bandwidth',color='orange')
plt.plot(x,PredY.bandwidth,'--',label='Predicted avg bandwidth',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Bandwidth')
plt.show()

plt.plot(x,TestY.ioread,'.',label='Test avg I/O Read',color='orange')
plt.plot(x,PredY.ioread,'--',label='Predicted avg I/O Read',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('I/O')
plt.show()

plt.plot(x,TestY.iowrite,'.',label='Test avg I/O Write',color='orange')
plt.plot(x,PredY.iowrite,'--',label='Predicted avg I/O Write',color='darkgreen')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('I/0')
plt.show()
