import tensorflow as tf
from setup import FNCData, create_vectors, pipeline_train
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
keep_prob_pl = tf.placeholder(tf.float32)


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


# define loss, gradients and optimizer
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v)
                    for v in tf_vars if 'bias' not in v.name]) * l2_alpha
loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, tr_stances_pl) + l2_loss)
optimiser = tf.train.AdamOptimizer(lr)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss_op, tf_vars), clip_ratio)
opt_op = optimiser.apply_gradients(zip(grads, tf_vars))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
opt_func = tf.train.AdamOptimizer(learn_rate)
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
num_epochs = 90

raw_train = FNCData('dataset/train_stances.csv', 'dataset/train_bodies.csv')
raw_dev = FNCData('dataset/dev_stances.csv', 'dataset/dev_bodies.csv')
raw_dev = FNCData('dataset/test_stances.csv', 'dataset/test_bodies.csv')

bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = create_vectors(raw_train, raw_dev, raw_test, lim_unigram=5000)
train_x, train_Y = pipeline_train(raw_train, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for epoch in range(num_epochs):
        total_loss = 0
        indices = list(range(n_train))
        r.shuffle(indices)

        for i in range(n_train // batch_size):
            batch_indices = indices[i * batch_size: (i + 1) * batch_size]
            batch_features = [train_x[i] for i in batch_indices]
            batch_stances = [train_Y[i] for i in batch_indices]

            batch_feed_dict = {tr_features_pl: batch_features, tr_stances_pl: batch_stances, keep_prob_pl: tr_keep_prob}
            _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
            total_loss += current_loss

        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
    # TODO implement batch training     

