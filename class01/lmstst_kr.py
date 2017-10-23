import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor

def reg_model():
	model = Sequential()
	model.add(Dense(2, input_dim=1, activation='relu'))
	model.add(Dense(1))
	
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


xt =np.array([1,2,3,4] , dtype='float64')
yt = np.array([0,-1,-2,-3] , dtype='float64')

estimator = KerasRegressor(build_fn=reg_model, epochs=200, batch_size=1 , verbose=1)

estimator.fit(xt, yt)

yp = estimator.predict(np.array([1.5]))
print(yp)
