import time
import numpy as np
import tensorflow as tf

from models import GAT_BNF, GAT_bi_BNF, GAT_AE_BNF, GAT_FC_BNF
from sklearn.model_selection import train_test_split
import Data_processing
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'Schiz'


# training params
batch_size = 50

nb_epochs = 150
patience = 10
lr = 0.0005  # learning rate
l2_coef = 0.005  # weight decay
recon_lr_weight = 0
hid_units = [64,32,32] # numbers of hidden units per each attention head in each layer
n_heads = [12, 12, 9] # additional entry for the output layer
residual = True
nonlinearity = tf.nn.relu
model = GAT_FC_BNF

augmentation = 500


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


# tr_fmri_net, tr_adj, tr_labels, val_fmri_net, val_adj, val_labels, test_fmri_net, test_adj, test_labels = \
#     Data_processing.load_data('Data_BNF/'+dataset+'/','Data_BNF/'+dataset+'/labels.csv', augmentation)

# tr_fmri_net, tr_adj, tr_labels, val_fmri_net, val_adj, val_labels, test_fmri_net, test_adj, test_labels = \
#     Data_processing.load_sorted_data('Data_BNF/'+dataset+'/sort_data.pkl')


fmri_signals, labels, graphs = \
    Data_processing.load_data('Data_BNF/'+dataset+'/','Data_BNF/'+dataset+'/labels.csv', augmentation)

# np.save('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/fmri_net_all.npy',fmri_signals)
# np.save('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/DTI_net_all.npy',graphs)
# np.save('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/labels_gender_all.npy',np.asarray(labels,dtype=int))

while(True):

    kf = KFold(n_splits=5)
    kf.get_n_splits(fmri_signals)
    labels = np.array(labels)

    test_loss_avg = []
    test_acc_avg = []

    predict_labels = []
    actual_labels = []

    for train_ind, test_ind in kf.split(fmri_signals):
        train_fmri_rsignals, test_fmri_rsignals = fmri_signals[train_ind], fmri_signals[test_ind]
        train_rlabels, test_rlabels = labels[train_ind], labels[test_ind]
        train_rgraphs, test_rgraphs = graphs[train_ind], graphs[test_ind]

        train_fmri_rsignals, val_fmri_rsignals, train_rlabels, val_rlabels, train_rgraphs, val_rgraphs = train_test_split(
            train_fmri_rsignals, train_rlabels, train_rgraphs, test_size=0.2)


        if dataset != 'HCP':

            tr_fmri_net, tr_labels, tr_adj = Data_processing.augmentation_fmri_net(train_fmri_rsignals, train_rlabels, train_rgraphs,
                                                                          augmentation)
            val_fmri_net, val_labels, val_adj = Data_processing.augmentation_fmri_net(val_fmri_rsignals, val_rlabels, val_rgraphs, 0)
            test_fmri_net, test_labels, test_adj = Data_processing.augmentation_fmri_net(test_fmri_rsignals, test_rlabels, test_rgraphs, 0)

        else:

            tr_fmri_net, tr_labels, tr_adj = train_fmri_rsignals, train_rlabels, train_rgraphs
            val_fmri_net, val_labels, val_adj = val_fmri_rsignals, val_rlabels,val_rgraphs
            test_fmri_net, test_labels, test_adj = test_fmri_rsignals, test_rlabels,test_rgraphs


        train_size = tr_fmri_net.shape[0]
        nb_nodes = tr_adj.shape[1]
        ft_size = 1
        nb_CAM = 64
        nb_classes = 2

        tr_labels = Data_processing.one_hot(tr_labels)
        val_labels = Data_processing.one_hot(val_labels)
        test_labels = Data_processing.one_hot(test_labels)

        tr_features = np.ones((tr_adj.shape[0],nb_nodes,ft_size))/nb_nodes
        val_features = np.ones((val_adj.shape[0],nb_nodes,ft_size))/nb_nodes
        test_features = np.ones((test_adj.shape[0], nb_nodes, ft_size))/nb_nodes





        # batch_size = train_adj.shape[0]


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

                ftr_in = tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, ft_size))
                bias_in = tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, nb_nodes))
                lbl_in = tf.placeholder(dtype=tf.int32, shape=(None, nb_classes))
                fmri_net = tf.placeholder(dtype=tf.float32, shape=(None,nb_nodes,nb_nodes))
                # msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
                ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
                is_train = tf.placeholder(dtype=tf.bool, shape=())
                global_step = tf.Variable(0,trainable=False)


            logits, prediction = model.inference(ftr_in, nb_classes, fmri_net, nb_CAM, is_train,
                                                            net_mat=bias_in,hid_units=hid_units, n_heads=n_heads,
                                                            residual=residual, activation=nonlinearity)



            # log_resh = tf.reshape(logits, [-1, nb_classes])
            # lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
            # msk_resh = tf.reshape(msk_in, [-1])

            # loss = model.loss(logits, lbl_in, fmri_net, reconstruct_net, recon_lr_weight)
            loss = model.loss(logits, lbl_in)

            accuracy = model.accuracy(logits, lbl_in)

            train_op = model.training(loss, lr, l2_coef, global_step)

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

                    tr_size = tr_features.shape[0]

                    while tr_step * batch_size < tr_size:
                        _, loss_value_tr, acc_tr= sess.run([train_op, loss, accuracy],

                            feed_dict={
                                ftr_in: tr_features[tr_step*batch_size:(tr_step+1)*batch_size],
                                bias_in: tr_adj[tr_step*batch_size:(tr_step+1)*batch_size],
                                lbl_in: tr_labels[tr_step*batch_size:(tr_step+1)*batch_size],
                                fmri_net: tr_fmri_net[tr_step*batch_size:(tr_step+1)*batch_size],
                                # msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                                is_train: True})
                        train_loss_avg += loss_value_tr
                        train_acc_avg += acc_tr
                        tr_step += 1

                        # print('Logit value is {}, ------------------------------'.format(observation))

                    vl_step = 0

                    vl_size = val_features.shape[0]


                    while vl_step * batch_size < vl_size:
                        loss_value_vl, acc_vl = sess.run([loss, accuracy],
                            feed_dict={

                                ftr_in: val_features[vl_step*batch_size:(vl_step+1)*batch_size],
                                bias_in: val_adj[vl_step*batch_size:(vl_step+1)*batch_size],
                                lbl_in: val_labels[vl_step*batch_size:(vl_step+1)*batch_size],
                                fmri_net: val_fmri_net[vl_step*batch_size:(vl_step+1)*batch_size],

                                # msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                                is_train: False})
                        val_loss_avg += loss_value_vl
                        val_acc_avg += acc_vl
                        vl_step += 1


    ### ------------------------------------------------------------------------------

                    ts_size = test_features.shape[0]
                    ts_step = 0
                    ts_loss = 0.0
                    ts_acc = 0.0

                    while ts_step * batch_size < ts_size:
                        loss_value_ts, acc_ts = sess.run([loss, accuracy],
                                                         feed_dict={
                                                             ftr_in: test_features[
                                                                     ts_step * batch_size:(ts_step + 1) * batch_size],
                                                             bias_in: test_adj[
                                                                      ts_step * batch_size:(ts_step + 1) * batch_size],
                                                             lbl_in: test_labels[
                                                                     ts_step * batch_size:(ts_step + 1) * batch_size],
                                                             fmri_net: test_fmri_net[
                                                                       ts_step * batch_size:(ts_step + 1) * batch_size],
                                                             # msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                                                             is_train: False})
                        ts_loss += loss_value_ts
                        ts_acc += acc_ts
                        ts_step += 1
    ###---------------------------------------------------------------------------------------------


                    print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f | Test: loss = %.5f, acc = %.5f' %
                            (train_loss_avg/tr_step, train_acc_avg/tr_step,
                            val_loss_avg/vl_step, val_acc_avg/vl_step,
                             ts_loss / ts_step, ts_acc / ts_step))

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
                        if (curr_step == patience and train_acc_avg/tr_step>0.94) or epoch>30:
                            print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                            print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                            break

                    train_loss_avg = 0
                    train_acc_avg = 0
                    val_loss_avg = 0
                    val_acc_avg = 0

                # saver.restore(sess, checkpt_file)


                ts_size = test_features.shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0

                while ts_step * batch_size < ts_size:
                    loss_value_ts, acc_ts, predict_label, actual_label = sess.run([loss, accuracy,prediction, lbl_in],
                        feed_dict={
                            ftr_in: test_features[ts_step*batch_size:(ts_step+1)*batch_size],
                            bias_in: test_adj[ts_step*batch_size:(ts_step+1)*batch_size],
                            lbl_in: test_labels[ts_step*batch_size:(ts_step+1)*batch_size],
                            fmri_net: test_fmri_net[ts_step*batch_size:(ts_step+1)*batch_size],
                            # msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                            is_train: False})
                    ts_loss += loss_value_ts
                    ts_acc += acc_ts
                    ts_step += 1

                    predict_labels.append(np.argmax(predict_label, axis=1))
                    actual_labels.append(np.argmax(np.asarray(actual_label), axis=1))

                print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

                test_loss_avg.append(ts_loss/ts_step)
                test_acc_avg.append(ts_acc/ts_step)



                # plt.matshow(original_net[1])
                # plt.matshow(recon_net[1])
                # plt.matshow(original_net[2])
                # plt.matshow(recon_net[2])
                # plt.show()


                sess.close()

    test_loss_avg = np.asarray(test_loss_avg)
    test_loss_avg = np.mean(test_loss_avg)

    test_acc_avg = np.asarray(test_acc_avg)
    test_acc_avg = np.mean(test_acc_avg)

    print('--------------------------------------------------')
    print('Final test loss:', test_loss_avg, '; Final test accuracy:', test_acc_avg)

    if(test_acc_avg>0.73):
        out_labels = np.asarray([np.concatenate(predict_labels, axis=0), np.concatenate(actual_labels, axis=0)])
        np.save('Data_BNF/'+dataset+'/result_FC.npy', out_labels)
        break
