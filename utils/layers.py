import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# input dim:
# seq: [batch_size, nb_nodes, nb_features, nb_slots]
# out: [batch_size, nb_nodes, nb_newfeatures, nb_newslots]
def attn_head_BNF(seq, out_sz, reweight_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        batch_size, nb_nodes, nb_features, nb_slots = seq.get_shape()
        seq_tmp = tf.transpose(seq,perm=[0,1,3,2])
        seq_tmp = tf.reshape(seq_tmp, shape=[-1,nb_nodes*nb_slots,nb_features])

        seq_fts = tf.layers.conv1d(seq_tmp, out_sz, 1, use_bias=False)

        seq_fts = tf.reshape(seq_fts, shape=[-1, nb_nodes, nb_slots, out_sz])
        seq_fts = tf.transpose(seq_fts, perm=[0,2,1,3])
        reweight_mat = tf.transpose(reweight_mat,perm=[0,3,1,2])  # [batch_size, nb_slots, nb_nodes, nb_nodes]
        # simplest self-attention possible

        vals = tf.matmul(reweight_mat, seq_fts)
        vals = tf.contrib.layers.bias_add(vals)
        vals = tf.transpose(vals, perm=[0,2,3,1])
        vals =activation(vals) # [batch_size, nb_nodes, nb_filters, nb_slots]

        ret = tf.layers.conv2d(vals, nb_slots, 1, activation=activation)
        ret = tf.nn.softmax(ret, axis=1)

        # residual connection
        if residual:
            # if seq.shape[-1] != ret.shape[-1]:
            #     ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            # else:
            #     seq_fts = ret + seq
            ret = tf.concat([ret,seq],axis=2)

        return  ret# activation


