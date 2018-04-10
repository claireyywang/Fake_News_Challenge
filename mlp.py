import tensorflow as tf

# Training Parameters 
batch_size = 500
tr_keep_prob = 0.6
l2_alpha = 0.00001
lr = 0.01
clip_ratio = 5

# Network Parameters
class_num = 4
hidden_num = 100
# TODO: change feature size
feature_size = len(features)

# Create placeholders: Inserts a placeholder for a tensor that will be always fed.
tr_features_pl = tf.placeholder(tf.float32, shape=[None, feature_size], name='features')
tr_stances_pl = tf.placeholder(tf.int64, shape=[None], name='stances')


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

logits = MLP(tr_features_pl)

# prediction
softmax_out = tf.nn.softmax(logits)
pred = tf.arg_max(softmax_out, 1)


# define loss and optimizer
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf_vars if 'bias' not in v.name]) * l2_alpha
loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, tr_stances_pl) + l2_loss)
optimiser = tf.train.AdamOptimizer(lr)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
opt_op = optimiser.apply_gradients(zip(grads, tf_vars))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # TODO implement batch training     

