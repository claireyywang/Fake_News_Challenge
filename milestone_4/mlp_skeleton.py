import random
import tensorflow as tf

# import all functions in setup module
from setup import *

tr_stances_file = '../dataset/train_stances.csv'
tr_bodies_file = '../dataset/train_bodies.csv'
dev_stances_file = '../dataset/dev_stances.csv'
dev_bodies_file = '../dataset/dev_bodies.csv'
test_stances_file = '../dataset/test_stances.csv'
test_bodies_file = '../dataset/test_bodies.csv'

dev_pred_file = 'output/dev_predictions.csv'
test_pred_file = 'output/test_predictions.csv'

print('loading data...')
train_data = FNCData(tr_stances_file, tr_bodies_file)
dev_data = FNCData(dev_stances_file, dev_bodies_file)
test_data = FNCData(test_stances_file, test_bodies_file)

bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = create_vectors(train_data, dev_data, test_data, lim_unigram=5000)

# extract features and labels
print('extracting features...')
train_X, train_y = pipeline_train(train_data, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
dev_X, dev_y = pipeline_dev(dev_data, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
text_X = pipeline_test(test_data, bow_vectorizer,tfreq_vectorizer, tfidf_vectorizer)

# Training Parameters 
batch_size = 500
tr_keep_prob = 0.6
l2_alpha = 0.00001
lr = 0.01
clip_ratio = 5

# Network Parameters
class_num = 4
hidden_num = 100

"""
TODO: change feature size

"""
tr_feature_size = len(train_X[0])

epochs = 90

# Create placeholders: Inserts a placeholder for a tensor that will be always fed.
features_pl = tf.placeholder(tf.float32, shape=[None, tr_feature_size], name='features')
stances_pl = tf.placeholder(tf.int64, shape=[None], name='stances')
kp_pl = tf.placeholder(tf.float32)

def MLP(features):
    # Initialize neural network structure 
    # define 100 hidden layer 
    hidden_out = tf.contrib.layers.fully_connected(features,hidden_num,activation_fn=tf.nn.relu)
    # compute dropout 
    dropout1 = tf.nn.dropout(hidden_out, tr_keep_prob)
    # linear layer, output 4 classifications 
    logits_flat = tf.contrib.layers.fully_connected(dropout1, class_num, activation_fn=None)
    # compute dropout 
    dropout2 = tf.nn.dropout(logits_flat, tr_keep_prob)
    return tf.reshape(dropout2, [batch_size, class_num])

# compute probabilities of four classes from mlp 
print('building neural net...')
logits = MLP(features_pl)

# prediction
softmax_out = tf.nn.softmax(logits)
pred = tf.argmax(softmax_out, 1)

# define loss, gradients and optimizer
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf_vars if 'bias' not in v.name]) * l2_alpha
loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)
optimiser = tf.train.AdamOptimizer(lr)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tf_vars), clip_ratio)
opt_op = optimiser.apply_gradients(zip(grads, tf_vars))

# split features and stances into dictionary format batches
# batches: a list of dicts
def make_batch(X_pl, y_pl, X, y, kp):
    feature_num = len(X)
    inds = list(range(feature_num))
    random.Random().shuffle(inds)
    batches = []

    for i in range(feature_num//batch_size):
        batch_inds = inds[i * batch_size: (i+1)* batch_size]
        batch_X = [X[i] for i in batch_inds]
        batch_y = [y[i] for i in batch_inds]

        batch_dict = {X_pl: batch_X, y_pl: batch_y, kp_pl: kp}
        batches.append(batch_dict)
    return batches

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # train the batches for the specified number of epochs
    for epoch in range(epochs):
    	"""
    	TODO: Implement batch training for the neural network. Use the make_batch function to
    		  create the appropriate batches from the features and labels defined above. In
    		  addition, make sure to keep track of the loss that occurs over each batch.

   		"""  
    
    # pred on dev 
    print('predicting on dev features...')
    dev_dict = {features_pl: dev_X, kp_pl:1.0}
    dev_pred = sess.run(pred, feed_dict=dev_dict)

    # pred on test 
    print('predicting on test features...')
    test_dict = {features_pl: text_X, kp_pl: 1.0}
    test_pred = sess.run(pred, feed_dict=test_dict)

save_predictions(dev_pred, dev_pred_file)
save_predictions(test_pred, test_pred_file)