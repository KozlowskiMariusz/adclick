
import tensorflow as tf
#import random_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

path = '/home/mariusz/adclick/train.csv'
path1 = '/home/mariusz/adclick/test.csv'

# start = time.clock()

clicks = pd.read_csv(path,nrows=1e6)
clicks_test = pd.read_csv(path1)

# end = time.clock()
# print(end - start)
# print(sum(df.is_attributed))

from sklearn.metrics import accuracy_score

train_X, test_X, train_y, test_y = train_test_split(clicks.iloc[:,:5],pd.DataFrame(clicks.is_attributed),test_size= 0.3, random_state=17278)

train_y = np.eye(2)[train_y.values.ravel()]
test_y = np.eye(2)[test_y.values.ravel()]

print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)

hm_epochs = 5
batch_size = 32

n_nodes_input = 5
n_nodes_hl1 = 30
n_nodes_hl2 = 20
n_nodes_hl3 = 10

n_classes = 2
#batch_size = 100

graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder('float', [None, n_nodes_input])
    y = tf.placeholder('float')

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    hl1 = tf.add(tf.matmul(x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    hl1 = tf.nn.relu(hl1)

    hl2 = tf.add(tf.matmul(hl1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    hl2 = tf.nn.relu(hl2)

    hl3 = tf.add(tf.matmul(hl2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    hl3 = tf.nn.relu(hl3)

    output = tf.matmul(hl3, output_layer['weights']) + output_layer['biases']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_X):
            start = i
            end = min(i+batch_size, len(train_X))
            batch_X = np.array(train_X[start:end])
            batch_y = np.array(train_y[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_X, y: batch_y})
            epoch_loss += c
            i += batch_size
        print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

    print('Accuracy_train:', accuracy.eval({x: train_X, y: train_y}))
    print('Accuracy_test:', accuracy.eval({x: test_X, y: test_y}))
