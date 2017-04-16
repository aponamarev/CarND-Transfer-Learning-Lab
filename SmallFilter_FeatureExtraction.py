#Load cifar10
from keras.datasets import cifar10
titles_list = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#provied labels contain extra dimention. To get rid of it, let's squeeze the dataset
import numpy as np
from matplotlib import pyplot as plt

y_train=y_train.reshape(-1)
y_test=y_test.reshape(-1)
#calculate the numer of classes
n_classes=y_train.max()+1

#print out the stats
print("train set shape:", X_train.shape)
print("test set shape:", X_test.shape)

n, bins, patches = plt.hist(y_train,bins=n_classes, color='blue')
plt.xlabel('classes')
plt.ylabel('samples')
plt.title("Samples distribution")
plt.axis([0,10,0,6000])
#plt.show()

#Load the metagraph of LeNet with batch normalization and dropout
import tensorflow as tf
from src.nets.SmallFilters import SmallFilters

X_ph = tf.placeholder(dtype=tf.float32, shape=(None, 32,32,3), name='X')
Y_ph = tf.placeholder(shape=(None,), name='labels', dtype=tf.int64)
net = SmallFilters(X_ph, Y_ph, 43)


sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
new_saver = tf.train.Saver(reshape=True)

#Restore the session and the data
last_checkpoint = tf.train.latest_checkpoint('logs/rebalanced_set/SmallFilters')
new_saver.restore(sess, last_checkpoint)

#Get the accuracy opeartion from the graph
probabilities = net.probability_op
top5_probabilities = tf.nn.top_k(probabilities, k=5, name="top5")

for i in range(10):
    img = X_train[i]
    minibatch = img[np.newaxis]
    top5_prob, top5_indeces = sess.run(top5_probabilities, feed_dict={net.X: minibatch,
                                                       net.dropout_keep_rate: 1.0,
                                                       net.is_training_mode:False})
    print("{} Images. Probabilities: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(titles_list[y_train[i]], *top5_prob))
    print("{} Images. Indecies: {},{},{},{},{}".format(titles_list[y_train[i]], *top5_indeces))



# Setup a new classifier for 10 classes (cifar10_classifier)
feature_map = tf.stop_gradient(net.feature_map)

with tf.name_scope("cifar10_classifier"):
    shape = [net._N_CLASSES, n_classes]
    W = tf.get_variable("kernel", shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("bias", shape=n_classes, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    cifar10_logits = tf.nn.xw_plus_b(feature_map, W, b, name="logits")
    cifar10_prob = tf.nn.softmax(cifar10_logits, name="probabilities")

# Define accuracy
with tf.name_scope("cifar10_accuracy"):
    cifar10_accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.arg_max(tf.nn.softmax(logits=cifar10_logits),1), Y_ph),
            dtype=tf.float32))

# Define loss (cifar10_loss)
with tf.name_scope('cifar10_loss'):
    cifar10_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cifar10_logits, labels=Y_ph))

# Define optimizer
opt = tf.train.AdamOptimizer()
minimize_op = opt.minimize(cifar10_loss)

# Execute the training
import tqdm, os
from sklearn.utils import shuffle
epocs = 7
batch_size = 512
log_path = 'logs/rebalanced_set/SmallFilters'

summary_op = tf.summary.merge_all()
summary_writter = tf.summary.FileWriter(os.path.join(os.path.join(log_path, "cifar10/"), "tensorboard/"),
                                        graph=sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(epocs):
    X_train, y_train = shuffle(X_train, y_train)
    pbar = tqdm.trange(0, X_train.shape[0], batch_size)
    for shift in pbar:
        shift_end = shift+batch_size
        minibatch_img = X_train[shift:shift_end]
        minibatch_indeces = y_train[shift:shift_end]
        processed_ops = [minimize_op, cifar10_loss, cifar10_accuracy]
        minibatch_feed_dict = {X_ph: minibatch_img,
                               Y_ph: minibatch_indeces,
                               net.is_training_mode: False,
                               net.dropout_keep_rate: 1.0}
        _, minibatch_loss, minibatch_acc = sess.run(processed_ops, feed_dict=minibatch_feed_dict)
        pbar.set_postfix(acc=minibatch_acc, loss=minibatch_loss)
    new_saver.save(sess, save_path=os.path.join(log_path, "cifar10/"))
    summary_writter.add_summary(sess.run(summary_op, feed_dict=minibatch_feed_dict), i)
print("Finished the training.")


print("Finish!")