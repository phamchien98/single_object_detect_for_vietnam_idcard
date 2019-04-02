
import pickle

# Getting back the objects:
with open('X.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X = pickle.load(f)
with open('y_label.pkl','rb') as f:  # Python 3: open(..., 'rb')
    y_label = pickle.load(f)


print(X)

print(type(X))

print(y_label)

print(type(y_label))

y_label.shape

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
X_train,X_validation,y_train,y_validation = train_test_split(X,y_label,test_size = 0.2,random_state = 42)

X_train.shape

y_train.shape

x_placeholder = tf.placeholder(tf.float32,[None,300,300,3])
y_placeholder = tf.placeholder(tf.float32,[None,4,2])

num_points = 4
conv1 = tf.layers.conv2d(x_placeholder,32,5,activation = tf.nn.relu,padding="SAME")
conv1 = tf.layers.max_pooling2d(conv1,2,2,padding = "SAME")
conv2 = tf.layers.conv2d(conv1,64,5,activation = tf.nn.relu,padding="SAME")
conv2 = tf.layers.max_pooling2d(conv2,2,2,padding = "SAME")
conv3 = tf.layers.conv2d(conv2,128,5,activation = tf.nn.relu,padding="SAME")
conv3 = tf.layers.max_pooling2d(conv3,2,2,padding = "SAME")
conv4 = tf.layers.conv2d(conv3,64,5,activation = tf.nn.relu,padding="SAME")
conv4 = tf.layers.max_pooling2d(conv4,2,2,padding = "SAME")
conv5 = tf.layers.conv2d(conv4,32,5,activation = tf.nn.relu,padding="SAME")
conv5 = tf.layers.max_pooling2d(conv5,2,2,padding = "SAME")


fc1 = tf.contrib.layers.flatten(conv5)
fc1 = tf.layers.dense(fc1,512,activation = tf.nn.relu)
out = tf.layers.dense(fc1,num_points * 2)
out=tf.reshape(out,[-1,4,2])

batch_size = 50
def random_batch(x_train,y_train,batch_size):
  rnd_indices = np.random.randint(0,len(x_train),batch_size)
  x_batch = x_train[rnd_indices]
  y_batch = y_train[rnd_indices]
  return x_batch,y_batch

pred = out
print(tf.shape(pred))
print(tf.shape(y_placeholder))

subtract = tf.math.subtract(pred,y_placeholder)
binhphuong = tf.math.multiply(subtract,subtract)
distance = tf.math.sqrt(binhphuong[:,:,0]+binhphuong[:,:,1])

loss = tf.math.reduce_mean(distance)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

num_steps = 500
n_epochs = 50
min_loss = 10000
for step in range(1,num_steps+1):
  x_batch,y_batch = random_batch(X_train,y_train,batch_size)
  loss_value,_ = sess.run([loss,training_op], feed_dict = {x_placeholder:x_batch ,y_placeholder:y_batch})
  print(loss_value)
  if loss_value < min_loss :
    min_loss = loss_value
    save_path = saver.save(sess, "./model.ckpt")
print("Optimization Finished!")

loss_value= sess.run([loss], feed_dict = {x_placeholder:X_validation,y_placeholder:y_validation})
print(loss_value)



img1 = cv2.imread('testcmt.jpg')
import copy
img1_copy = copy.deepcopy(img1)

img1.shape
old_shape = img1.shape
ti_le = [old_shape[1]/300,old_shape[0]/300]

img1 = cv2.resize(img1,(300,300))

pred_value= sess.run([pred], feed_dict = {x_placeholder:[img1]})[0][0]

for point in pred_value:
  print([int(point[0]*ti_le[0]),int(point[1]*ti_le[1])])
  cv2.circle(img1_copy, (int(point[0]*ti_le[0]),int(point[1]*ti_le[1])), 5, (0, 0, 255), -1)

import matplotlib.pyplot as plt

plt.imshow(img1_copy)
plt.show()

tf.reset_default_graph()
x_placeholder = tf.placeholder(tf.float32,[None,300,300,3])
y_placeholder = tf.placeholder(tf.float32,[None,4,2])

num_points = 4
conv1 = tf.layers.conv2d(x_placeholder,32,5,activation = tf.nn.relu,padding="SAME")
conv1 = tf.layers.max_pooling2d(conv1,2,2,padding = "SAME")
conv2 = tf.layers.conv2d(conv1,64,5,activation = tf.nn.relu,padding="SAME")
conv2 = tf.layers.max_pooling2d(conv2,2,2,padding = "SAME")
conv3 = tf.layers.conv2d(conv2,128,5,activation = tf.nn.relu,padding="SAME")
conv3 = tf.layers.max_pooling2d(conv3,2,2,padding = "SAME")
conv4 = tf.layers.conv2d(conv3,64,5,activation = tf.nn.relu,padding="SAME")
conv4 = tf.layers.max_pooling2d(conv4,2,2,padding = "SAME")
conv5 = tf.layers.conv2d(conv4,32,5,activation = tf.nn.relu,padding="SAME")
conv5 = tf.layers.max_pooling2d(conv5,2,2,padding = "SAME")


fc1 = tf.contrib.layers.flatten(conv5)
fc1 = tf.layers.dense(fc1,512,activation = tf.nn.relu)
out = tf.layers.dense(fc1,num_points * 2)
out=tf.reshape(out,[-1,4,2])

pred = out

subtract = tf.math.subtract(pred,y_placeholder)
binhphuong = tf.math.multiply(subtract,subtract)
distance = tf.math.sqrt(binhphuong[:,:,0]+binhphuong[:,:,1])

loss = tf.math.reduce_mean(distance)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

# Initializing the variables
sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "./model.ckpt")



img1 = cv2.imread('testcmt.jpg')
import copy
img1_copy = copy.deepcopy(img1)
img1.shape
old_shape = img1.shape
ti_le = [old_shape[1]/300,old_shape[0]/300]
img1 = cv2.resize(img1,(300,300))
pred_value= sess.run([pred], feed_dict = {x_placeholder:[img1]})[0][0]
for point in pred_value:
  print([int(point[0]*ti_le[0]),int(point[1]*ti_le[1])])
  cv2.circle(img1_copy, (int(point[0]*ti_le[0]),int(point[1]*ti_le[1])), 5, (0, 0, 255), -1)
import matplotlib.pyplot as plt

plt.imshow(img1_copy)
plt.show()

