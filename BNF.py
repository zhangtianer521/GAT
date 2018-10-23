import time
import numpy as np
import tensorflow as tf

from models import GAT, GAT_BNF
from utils import process
import Data_processing

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'Schiz'


# training params
batch_size = 100
nb_epochs = 10000
patience = 50
lr = 0.01  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [16,32,32,16] # numbers of hidden units per each attention head in each layer
n_heads = [2, 2, 2,2] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.relu
model = GAT_BNF

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

train_features, train_adj, train_labels, test_features, test_adj, test_labels = \
    Data_processing.load_data('Data_BNF/'+dataset+'/','Data_BNF/'+dataset+'/labels.csv')
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
# train_features, spars = process.preprocess_features(train_features)

nb_nodes = train_features.shape[1]
ft_size = train_features.shape[2]
nb_classes = train_labels.shape[1]
nb_slot = 5

batch_size = train_features.shape[0]

# adj = adj.todense()

# train_features = train_features[...,np.newaxis]
# test_features = test_features[...,np.newaxis]
# train_adj = train_adj[..., np.newaxis]
# test_adj = test_adj[..., np.newaxis]

# y_train = y_train[np.newaxis]
# y_val = y_val[np.newaxis]
# y_test = y_test[np.newaxis]
# train_mask = train_mask[np.newaxis]
# val_mask = val_mask[np.newaxis]
# test_mask = test_mask[np.newaxis]

# biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, ft_size, nb_slot))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, nb_nodes, nb_slot))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(None, nb_classes))
        # msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits, prediction = model.inference(ftr_in, nb_classes, nb_nodes, nb_slot, is_train,
                                attn_drop, ffd_drop,
                                reweight_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    # log_resh = tf.reshape(logits, [-1, nb_classes])
    # lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    # msk_resh = tf.reshape(msk_in, [-1])
    loss = model.loss(logits, lbl_in)
    accuracy = model.accuracy(logits, lbl_in)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = train_features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr, logits_val = sess.run([train_op, loss, accuracy, logits],

                    feed_dict={
                        ftr_in: train_features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: train_adj[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: train_labels[tr_step*batch_size:(tr_step+1)*batch_size],
                        # msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

                # print('Logit value is {}, ------------------------------'.format(logits_val))

            vl_step = 0
            vl_size = test_features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ftr_in: test_features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: test_adj[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: test_labels[vl_step*batch_size:(vl_step+1)*batch_size],
                        # msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        # ts_size = features.shape[0]
        # ts_step = 0
        # ts_loss = 0.0
        # ts_acc = 0.0
        #
        # while ts_step * batch_size < ts_size:
        #     loss_value_ts, acc_ts = sess.run([loss, accuracy],
        #         feed_dict={
        #             ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
        #             bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
        #             lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
        #             msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
        #             is_train: False,
        #             attn_drop: 0.0, ffd_drop: 0.0})
        #     ts_loss += loss_value_ts
        #     ts_acc += acc_ts
        #     ts_step += 1
        #
        # print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
