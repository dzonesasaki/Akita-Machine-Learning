import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]

estimater = tf.contrib.learn.DNNRegressor(
	feature_columns=feature_columns,
	hidden_units=[2])

xt ={'x': np.array([1,2,3,4] , dtype='float64')}
yt = np.array([0,-1,-2,-3] , dtype='float64')

input_fn_train = tf.estimator.inputs.numpy_input_fn(x=xt,y=yt,batch_size=1,num_epochs=1,shuffle=False)
estimater.fit(input_fn = input_fn_train ,steps=4)

input_fn_est2 = tf.estimator.inputs.numpy_input_fn(x={'x':np.array([1.5])}, batch_size=1 , shuffle=False,num_epochs=1)

yL=estimater.predict( input_fn=input_fn_est2 )
yp=list(estimater.predict(1.5))

print(yp)
