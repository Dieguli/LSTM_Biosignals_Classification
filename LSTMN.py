import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.tools import freeze_graph


# PREPARING DATA AND PREPROCESSING
columns = ['channel_1','channel_2','channel_3', 'channel_4']


#Three classes model
class1_data = pd.read_csv('class1_200_notch.txt',sep = " ", header = None, names = columns)
class2_data = pd.read_csv('class3_200_notch.txt',sep = " ", header = None, names = columns)
class3_data = pd.read_csv('class4_200_notch.txt',sep = " ", header = None, names = columns)



samples = 200
features = 4 
step = 200
samples_array = []
targets = []
print(len(class1_data)) 

for i in range(0, len(class1_data) - 199, step):
    ch1 = class1_data['channel_1'].values[i: i + samples]
    ch2 = class1_data['channel_2'].values[i: i + samples]
    ch3 = class1_data['channel_3'].values[i: i + samples]
    ch4 = class1_data['channel_4'].values[i: i + samples]
    samples_array.append([ch1, ch2, ch3, ch4])
    y= [1.0, 0.0, 0.0]
    targets.append(y)


for i in range(0, len(class1_data) - 199, step):
    ch1 = class2_data['channel_1'].values[i: i + samples]
    ch2 = class2_data['channel_2'].values[i: i + samples]
    ch3 = class2_data['channel_3'].values[i: i + samples]
    ch4 = class2_data['channel_4'].values[i: i + samples]
    samples_array.append([ch1, ch2, ch3, ch4])
    y= [0.0, 1.0, 0.0]
    targets.append(y)

for i in range(0, len(class1_data) - 199, step):
    ch1 = class3_data['channel_1'].values[i: i + samples]
    ch2 = class3_data['channel_2'].values[i: i + samples]
    ch3 = class3_data['channel_3'].values[i: i + samples]
    ch4 = class3_data['channel_4'].values[i: i + samples]
    samples_array.append([ch1, ch2, ch3, ch4])
    y= [0.0, 0.0, 1.0]
    targets.append(y)

print(np.shape(samples_array))
data = np.asarray(samples_array, dtype= np.float32).reshape(-1, samples, features)
print(data.shape)

data = np.asarray(samples_array, dtype= np.float32).reshape(-1, samples, features)
print(data.shape)

seed = 42
targets = np.array(targets)
print(targets.shape)

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=seed)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#------------------------------------------

samples = 200
features = 4 
classes= 3 
neurons = 70 

def create_LSTM_model(inputs):
    W = {
        'output': tf.Variable(tf.random_normal([neurons, classes]))
    }
    biases = {
        'output': tf.Variable(tf.random_normal([classes]))
    }

    x = tf.unstack(input, samples, 1)

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(neurons, forget_bias=1.0) for _ in range(1)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, x, dtype=tf.float32)
    lstm_last_output = outputs[-1]

    

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, samples, features], name="input")
# [time_steps, batch, num_features]
Y = tf.placeholder(tf.float32, [None, classes])

y_pred = create_LSTM_model(X)
y_softmax = tf.nn.softmax(y_pred, name="y_")

#using L2 regularization for minimizing the loss

l = 0.0015
L2 = l * sum(tf.nn.l(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, targets= Y)) + L2

#Defining the optimizer for the model

eta = 0.002

optimizer = tf.train.AdamOptimizer(eta=eta).minimize(loss)

correct_pred = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

#Training the model

results = dict(training_loss = [], training_accuracy = [], test_loss = [], test_accuracy = [])

epochs = 80
batch = 20

total_samples =len(x_train)
print
total_samples)

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

total_samples = len(x_train) #number of rows
	for i in range(1, epochs + 1):
		for start, end in zip(range(0,
            total_samples, batch), range(batch,
        total_samples + 1, batch)):
			sess.run(optimizer, feed_dict={X:x_train[start:end],
                                       Y:y_train[start:end]})
		_, acc_train, loss_train = sess.run([y_softmax, accuracy, loss], feed_dict={X: x_train, Y:y_train})
		_, acc_test, loss_test = sess.run([y_softmax, accuracy, loss], feed_dict={X: x_test, Y:y_test})
		results['training_loss'].append(loss_train)
		results['training_accuracy'].append(acc_train)
		results['test_loss'].append(loss_test)
		results['test_accuracy'].append(acc_test)
		print("test accuracy in results {0:f}".format(acc_test))
		print("test loss in results {0:f}".format(loss_test))
		print("epoch = " + str(i))
	predictions, acc_final, loss_final = sess.run([y_softmax, accuracy, loss], feed_dict={X: x_test, Y:y_test})
	print()
	print("Final Results: Accuracy: {0:.2f}, Loss: {1:.2f}".format(acc_final,loss_final))

	print("training finished ") 
	plt.plot(results['training_loss'], 'k-', label='Train Loss')
	plt.plot(results['test_loss'], 'r--', label='Test Loss')
	plt.title('Loss (MSE) per Epoch')
	plt.legend(loc='upper right')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.show()
