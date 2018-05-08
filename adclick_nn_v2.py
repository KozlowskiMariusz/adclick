import tensorflow as tf
import numpy as np
import pandas as pd 
import random
import argparse
from sklearn.model_selection import train_test_split
from additional_functions import xavier_init

parser = argparse.ArgumentParser()

parser.add_argument('-layers',required=True, help = 'architecture of neural network', type=int, nargs='+')
parser.add_argument('-epochs', '--hm_epochs', help="number of epochs", type=int, default =3000)
parser.add_argument('-checkpoint', '--hm_epochs_print', action="store", help="number of epochs print - loss value checkpoint",type=int, default=1000)
parser.add_argument('-alpha', required=True, help="learning rate", type=float)
parser.add_argument('-batch', '--batch_size', action="store", help="batch size", type=int,  default=5000)
parser.add_argument('-teta', help="l1 + l2 penalization - tuning parameter from interval [0,1], teta=0 (ridge), teta=1 (lasso)", type=float, default = 0)
parser.add_argument('-beta', help="l1 + l2 penalization - tuning parameter", type=float, default=0)
parser.add_argument('-seed', help="random seed for weights", type=float, default = 0)
parser.add_argument('-drop_out', help="Probability for drop-out", type=float, default = 1.0)

args = parser.parse_args()

# Reading data frame


path = '/home/mariusz/adclick/train.csv'
path1 = '/home/mariusz/adclick/test.csv'
# start = time.clock()

clicks = pd.read_csv(path,nrows=1e7)
clicks_test = pd.read_csv(path1)

# end = time.clock()
# print(end - start)
# print(sum(df.is_attributed))

from sklearn.metrics import accuracy_score

train_x, test_x, train_y, test_y = train_test_split(clicks.iloc[:,:5],pd.DataFrame(clicks.is_attributed),test_size= 0.3, random_state=17278)


print('trivial accuracy_train: ',  1- train_y.values.sum()/ len(train_y.values))
print('trivial accuracy_test: ',  1-  train_y.values.sum()/ len(train_y.values))

weight_coeff = len(train_y)/train_y.values.sum()


train_y = pd.DataFrame( np.eye(2)[train_y.values.ravel()])
test_y = pd.DataFrame( np.eye(2)[test_y.values.ravel()])

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


n_classes = 2 # continous output
n_variables = 5

#defining arguments given by user

alpha = args.alpha
batch_size = args.batch_size
hm_epochs = args.hm_epochs
hm_epochs_print = args.hm_epochs_print
p = train_x.shape[1]
print("train_x dimensions: ", train_x.shape)
layers = [p] + args.layers + [n_classes]
n_layers = len(layers)
print("layers: ", layers)
beta = args.beta
teta = args.teta
seed = args.seed
prob = args.drop_out
#prob = 1.0


graph = tf.Graph()

with graph.as_default():

    x = tf.placeholder(tf.float32,shape=[None,n_variables])
    y = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
	
    def train_neural_network(data):
        last_hidden_layer = data
        l1 = tf.constant(0,dtype=tf.float32)
        l2 = tf.constant(0,dtype=tf.float32)
        #tf.set_random_seed(seed)
        tf.set_random_seed(seed)
        for j in range(2,n_layers): 
            previous_layer = last_hidden_layer
            xavier_stddev =  xavier_init([layers[j-2], layers[j-1]] )
            weight  = tf.Variable(tf.random_normal([layers[j-2], layers[j-1] ], stddev=xavier_stddev )) 
            bias = tf.Variable(tf.random_normal([layers[j-1]], stddev=xavier_stddev))
            last_hidden_layer = tf.nn.relu( tf.add(tf.matmul(previous_layer, weight), bias))
            drop_out = tf.nn.dropout(last_hidden_layer,keep_prob)# keep_prob)
            l1 = l1 + tf.norm(bias) + tf.norm(weight)
            
            #l2 = l2 + tf.square(tf.norm(bias)) + tf.square(tf.norm(weight))         
            l2 = l2 + tf.nn.l2_loss(bias) + tf.nn.l2_loss(weight)


        xavier_stddev =  xavier_init([layers[-2], layers[-1]] )
        weight_output = tf.Variable(tf.random_normal([layers[-2], layers[-1]], stddev=xavier_stddev))
        bias_output =  tf.Variable(tf.random_normal([layers[-1]], stddev=xavier_stddev))
        output = tf.add(tf.matmul(drop_out, weight_output), bias_output)
        #output = tf.transpose(output)
        l1 = l1 + tf.norm(weight_output) + tf.norm(bias_output)
	#l2 = l2 + tf.square(tf.norm(bias_output)) + tf.square(tf.norm(weight_output))         
        l2 = l2 + tf.nn.l2_loss(bias_output) + tf.nn.l2_loss(weight_output)        
        return output,l1,l2

    prediction, l1, l2 = train_neural_network(x)



    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) + beta*( 0.5*(1-teta)*l2 + teta*l1)
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=prediction, onehot_labels=y)) + beta*( 0.5*(1-teta)*l2 + teta*l1)
    
    #cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(prediction,y,pos_weight = weight_coeff)) + beta*( 0.5*(1-teta)*l2 + teta*l1)
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

    saver = tf.train.Saver()


    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
#tf.ConfigProto(device_count = {'GPU':0})
#with tf.Session(config=config) as sess:

#saver = tf.train.Saver()

with tf.Session(graph =graph) as sess:
    sess.run(tf.global_variables_initializer())
    #saver = tf.train.import_meta_graph('model.ckpt.meta')
    #saver.restore(sess, "model.ckpt")
    for epoch in range(hm_epochs):
        epoch_loss = 0
        i=0
        
        train_x = train_x.sample(frac=1,random_state=epoch).reset_index(drop=True)
        train_y = train_y.sample(frac=1,random_state=epoch).reset_index(drop=True)
        while i < len(train_x):
            
            start = i
            end = min(i+batch_size, len(train_x))
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            epoch_loss += c
            i+=batch_size
        
        if epoch % hm_epochs_print == 0:

            print('Accuracy_train', accuracy.eval({x: train_x,y: train_y,keep_prob: 1.0}), 'Accuracy_test', accuracy.eval({x: test_x, y: test_y,keep_prob: 1.0}))
            #mae = tf.reduce_mean(tf.abs(prediction-y))
            #print("epoch_loss:", epoch_loss)
            #print('mae train:', mae.eval({x:train_x, y:train_y,keep_prob: 1.0}), '  mae test:', mae.eval({x:test_x, y:test_y,keep_prob:1.0}) )
             
            #print('mae train:', mae.eval({x:train_x, y:train_y}), '  mae test:', mae.eval({x:test_x, y:test_y}) )
            #model_path =  '~/tensorflow/' +'fw' + '_' + ','.join(str(x) for x in layers) + ".ckpt"
            #model_path = "~/tensorflow/model.ckpt"
            save_path = saver.save(sess,"model.ckpt")

#c#confiionfig
   # print ("Model saved in file: %s" % save_path)

#Converting final tensors to pandas and extracting predictions

#    pred_train = pd.DataFrame(prediction.eval({x:train_x, y:train_y,keep_prob:1.0}))
#    pred_test = pd.DataFrame(prediction.eval({x:test_x, y:test_y, keep_prob:1.0}))
#
#    pred_train = pred_train.transpose()
#    pred_test = pred_test.transpose()
#
#    pred_train.columns = ['y_pred']
#    #train.reset_index(drop=True,inplace=True)
#    pred_train['y'] =  train_y
#    pred_train['error'] = pred_train['y_pred'] - pred_train['y']
#    pred_train['abs_error'] = abs( pred_train['y_pred'] - pred_train['y'])
#    mae_train=pred_train['abs_error'].mean()
#    print('mae_train',mae_train)
#
#    pred_test.columns = ['y_pred']
#    #test.reset_index(drop=True,inplace=True)
#    pred_test['y'] = test_y
#    pred_test['error'] = pred_test.ix[:,0] - pred_test['y']
#    pred_test['abs_error'] = abs( pred_test['y_pred'] - pred_test['y'])
#    mae_test = pred_test['abs_error'].mean()
#    print('mae_test', mae_test)
#
#    pred_test.to_csv('dnageTest.csv',sep=",",index=False)
#    pred_train.to_csv('dnageTrain.csv',sep=",",index=False)



