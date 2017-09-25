import tensorflow as tf


xt = [1,2,3,4]
yt = [0,-1,-2,-3]

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

a=tf.Variable([0.3],dtype=tf.float32)
b=tf.Variable([-0.3],dtype=tf.float32)

ye = a*x+b

pow2_delta = tf.square(ye-y)
loss = tf.reduce_sum(pow2_delta)

rateA=0.01
optimizer = tf.train.GradientDescentOptimizer(rateA)
train= optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

Nloop=10000

for k in range(Nloop):
	sess.run(train, {x:xt, y:yt })

print(sess.run([a,b]))
print(sess.run(ye,{x:1.5}))
