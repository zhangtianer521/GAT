import tensorflow as tf
import tensorflow.contrib.metrics as metrics

class BaseGAttN:

    def loss(logits, labels):

        # sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        # xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits), sample_wts)
        # return tf.reduce_mean(xentropy, name='xentropy_mean')

        classify_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # Recons_loss = tf.reduce_mean(tf.square(fmri_net-recon_net))
        # Recons_loss = tf.reduce_sum(Recons_loss,axis=1)
        return tf.reduce_mean(classify_loss, axis=0)

    def bi_loss(logits, labels, fmri_net, recon_net_fmri, DTI_net, recon_net_DTI, recon_lr_weight):

        # sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        # xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits), sample_wts)
        # return tf.reduce_mean(xentropy, name='xentropy_mean')

        classify_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        Recons_loss_fmri = tf.reduce_mean(tf.square(fmri_net-recon_net_fmri))
        Recons_loss_DTI = tf.reduce_mean(tf.square(DTI_net - recon_net_DTI))
        # Recons_loss = tf.reduce_sum(Recons_loss,axis=1)
        return tf.reduce_mean(classify_loss, axis=0)+recon_lr_weight*(Recons_loss_fmri+0.1*Recons_loss_DTI)

    def AE_classify_loss(logits, labels, fmri_net, recon_net_fmri, DTI_net, reconstruct_DTI, recon_lr_weight):

        classify_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        Recons_loss_DTI = tf.reduce_mean(tf.multiply(tf.square(DTI_net-reconstruct_DTI),tf.abs(DTI_net)))
        # Recons_loss_fmri = tf.reduce_mean(tf.multiply(tf.square(fmri_net-recon_net_fmri),tf.abs(fmri_net)))

        # Recons_loss_DTI = tf.reduce_mean(tf.square(DTI_net-reconstruct_DTI))
        Recons_loss_fmri = tf.reduce_mean(tf.square(fmri_net-recon_net_fmri))

        return 1*tf.reduce_mean(classify_loss, axis=0)+recon_lr_weight*(Recons_loss_fmri + 0.1*Recons_loss_DTI)

    def AE_regression_loss(logits, labels, fmri_net, recon_net_fmri, DTI_net, reconstruct_DTI, recon_lr_weight):

        # sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        # xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits), sample_wts)
        # return tf.reduce_mean(xentropy, name='xentropy_mean')

        classify_loss = tf.losses.mean_squared_error(labels, logits)
        Recons_loss_DTI = tf.reduce_mean(tf.multiply(tf.square(DTI_net - reconstruct_DTI), tf.abs(DTI_net)))
        Recons_loss_fmri = tf.reduce_mean(tf.multiply(tf.square(fmri_net - recon_net_fmri), tf.abs(fmri_net)))

        # Recons_loss_DTI = tf.reduce_mean(tf.square(DTI_net - reconstruct_DTI))
        # Recons_loss_fmri = tf.reduce_mean(tf.square(fmri_net - recon_net_fmri))

        # Recons_loss = tf.reduce_sum(Recons_loss,axis=1)
        return classify_loss +recon_lr_weight*(Recons_loss_fmri + Recons_loss_DTI)


    def syn_loss(logits, labels, fmri_net, recon_net_fmri, recon_lr_weight):

        # sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        # xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits), sample_wts)
        # return tf.reduce_mean(xentropy, name='xentropy_mean')

        classify_loss = tf.losses.mean_squared_error(labels, logits)
        # Recons_loss_fmri = tf.reduce_mean(tf.multiply(tf.square(fmri_net-recon_net_fmri),tf.abs(fmri_net)))
        Recons_loss_fmri = tf.reduce_mean(tf.square(fmri_net-recon_net_fmri))
        # Recons_loss = tf.reduce_sum(Recons_loss,axis=1)
        return classify_loss +recon_lr_weight*(Recons_loss_fmri)


    def training(loss, lr, l2_coef, global_step):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        current_lr = tf.train.exponential_decay(lr,global_step,10, 0.95, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate=current_lr)

        # training op
        train_op = opt.minimize(loss+lossL2)
        
        return train_op

    def accuracy(preds, labels):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

    def syn_accuracy(preds, labels):
        preds = tf.squeeze(preds)
        labels = tf.squeeze(labels)
        preds_mean = tf.reduce_mean(preds)
        labels_mean = tf.reduce_mean(labels)
        nominator = tf.reduce_sum((preds-preds_mean)*(labels-labels_mean))
        denominator = tf.sqrt(tf.reduce_sum(tf.square(preds-preds_mean))*tf.reduce_sum(tf.square(labels-labels_mean)))
        corref = nominator/(denominator+tf.keras.backend.epsilon())


        # predss = tf.cast(predss,dtype=tf.int32)
        # corref, _ = metrics.streaming_covariance(predss,labels)
        return corref

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)


    def MSE(preds, labels):
        preds = tf.squeeze(preds)
        labels = tf.squeeze(labels)
        return tf.reduce_mean(tf.square(labels-preds))

    def MAE(preds, labels):
        preds = tf.squeeze(preds)
        labels = tf.squeeze(labels)
        return tf.reduce_mean(tf.abs(labels - preds))






##########################
# Adapted from tkipf/gcn #
##########################

    # def masked_softmax_cross_entropy(logits, labels, mask):
    #     """Softmax cross-entropy loss with masking."""
    #     loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    #     mask = tf.cast(mask, dtype=tf.float32)
    #     mask /= tf.reduce_mean(mask)
    #     loss *= mask
    #     return tf.reduce_mean(loss)
    #
    # def masked_sigmoid_cross_entropy(logits, labels, mask):
    #     """Softmax cross-entropy loss with masking."""
    #     labels = tf.cast(labels, dtype=tf.float32)
    #     loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    #     loss=tf.reduce_mean(loss,axis=1)
    #     mask = tf.cast(mask, dtype=tf.float32)
    #     mask /= tf.reduce_mean(mask)
    #     loss *= mask
    #     return tf.reduce_mean(loss)

    # def masked_accuracy(logits, labels, mask):
    #     """Accuracy with masking."""
    #     correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    #     accuracy_all = tf.cast(correct_prediction, tf.float32)
    #     mask = tf.cast(mask, dtype=tf.float32)
    #     mask /= tf.reduce_mean(mask)
    #     accuracy_all *= mask
    #     return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels):
        """Accuracy with masking."""
        predicted = tf.argmax(tf.nn.softmax(logits),1)
        labels = tf.argmax(labels,1)

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)

        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1))
        fp = tf.count_nonzero(predicted * (labels - 1))
        fn = tf.count_nonzero((predicted - 1) * labels)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure, precision, recall
