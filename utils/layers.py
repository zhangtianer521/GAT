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
def attn_head_BNF(seq, out_sz, reweight_mat, activation, in_drop=0.0, residual=False):
    with tf.name_scope('my_attn_BNF'):
        # binomial = tf.Variable()

        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        batch_size, nb_nodes, nb_features, nb_slots = seq.get_shape()

        # seq_tmp = tf.layers.batch_normalization(seq)
        seq_tmp = tf.transpose(seq,perm=[0,1,3,2])


        seq_fts = tf.layers.conv2d(seq_tmp, out_sz, 1, use_bias=True)
        seq_fts = activation(seq_fts)


        seq_fts = tf.transpose(seq_fts, perm=[0,2,1,3])
        reweight_mat = tf.transpose(reweight_mat,perm=[0,3,1,2])  # [batch_size, nb_slots, nb_nodes, nb_nodes]
        # simplest self-attention possible

        vals = tf.matmul(reweight_mat, seq_fts)
        vals = tf.contrib.layers.bias_add(vals)
        vals = tf.transpose(vals, perm=[0,2,3,1])

        # vals = tf.layers.batch_normalization(vals)
        vals =activation(vals) # [batch_size, nb_nodes, nb_filters, nb_slots]



        ret = tf.layers.conv2d(vals, nb_slots, 1)



        # residual connection
        if residual:
            if seq.shape[-2] != ret.shape[-2]:
                seq_jump = tf.transpose(seq,perm=[0,1,3,2])
                seq_jump = tf.layers.conv2d(seq_jump,out_sz,1)
                seq_jump = tf.transpose(seq_jump,perm=[0,1,3,2])
                ret = ret+seq_jump

        ret = activation(ret)


        ret = tf.layers.conv2d(ret, nb_slots, 1, activation=activation)

        ret = tf.nn.l2_normalize(ret,axis=1)

        # residual connection
        # if residual:
        #     # if seq.shape[-1] != ret.shape[-1]:
        #     #     ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
        #     # else:
        #     #     seq_fts = ret + seq
        #     ret = tf.concat([ret,seq],axis=2)


        return  ret# activation


# input dim:
# net: [batch_size, nb_nodes, nb_nodes]
# out: [batch_size, nb_nodes, nb_nodes, nb_slots]
def sparsity_BNF(net, out_sz, residual=False):
    with tf.name_scope('my_sparsity_BNF'):
        net_tmp = tf.expand_dims(net, axis=-1)
        # net_tmp = tf.layers.batch_normalization(net_tmp)
        net_tmp= tf.layers.conv2d(net_tmp,out_sz,1,activation=tf.nn.tanh,use_bias=False)
        net_out = tf.multiply(tf.sign(net_tmp)+1,1)
        return net_out


#------------------------------------------------------------
# implementation of e2v and v2e
#------------------------------------------------------------
def attn_e2v_BNF(seq, layer_id, out_sz, n_heads, net_mat, activation, in_drop=0.0, residual=False):
    with tf.name_scope('my_AE_BNF'):
        #seq : [batch_size, nb_nodes, features]
        #net_mat: [batch_size, nb_nodes, nb_nodes]
        [batch_size, nb_nodes, features] = seq.get_shape().as_list()
        # inpulse1 = tf.get_variable("inpulse1_ly"+str(layer_id),shape=[batch_size, nb_nodes],trainable=True)
        # impulse1 = tf.layers.conv1d(seq,nb_nodes,[nb_nodes])

        aggall = []
        # net_mat_1 = tf.reshape(net_mat, shape=[-1, nb_nodes, 1, 1])
        # net_mat_1 = tf.layers.conv2d(net_mat_1, 1, kernel_size=[nb_nodes, 1])
        # net_mat_1 = tf.reshape(net_mat_1,shape=[-1,nb_nodes,nb_nodes])
        stimulate_W_1 = tf.Variable(tf.ones([nb_nodes,1]),trainable=True,name='stimulate_W_1')
        net_mat_1 = net_mat*stimulate_W_1
        stimulate_mag_1 = tf.Variable(tf.ones([1]),trainable=True,name='stimulate_mag_1')
        agg1 = tf.matmul(net_mat_1+tf.multiply(tf.eye(nb_nodes,nb_nodes),stimulate_mag_1),seq)
        # agg1 = tf.matmul(net_mat_1 , seq)
        agg1 = tf.layers.conv1d(agg1, out_sz, 1, activation=activation)
        aggall.append(agg1)
        # inpulse2 = tf.get_variable("inpulse2_ly" + str(layer_id), shape=[batch_size, nb_nodes], trainable=True)
        for i in range(1,n_heads):
            stimulate_W_1 = tf.Variable(tf.random_normal([nb_nodes, 1]), trainable=True, name='stimulate_W_'+str(i+1))
            net_mat_1 = net_mat * stimulate_W_1
            stimulate_mag_1 = tf.Variable(tf.random_normal([1]), trainable=True, name='stimulate_mag_'+str(i+1))
            agg2 = tf.matmul(net_mat_1+tf.multiply(tf.eye(nb_nodes,nb_nodes),stimulate_mag_1),agg1)
            # agg2 = tf.matmul(net_mat_1,agg1)
            agg2 = tf.layers.conv1d(agg2, out_sz, 1, activation=activation)
            aggall.append(agg2)
            agg1=agg2
        # agg3 = tf.matmul(net_mat+tf.eye(nb_nodes,nb_nodes),agg2)
        # agg3 = tf.layers.conv1d(agg3, out_sz, 1, activation=activation)
        # agg4 = tf.matmul(net_mat+tf.eye(nb_nodes,nb_nodes), agg3)
        # agg4 = tf.layers.conv1d(agg4, out_sz, 1, activation=activation)
        aggall = tf.concat(aggall,axis=-1)
        # aggall = tf.stack([agg1,agg2,agg3,agg4],axis=-1)
        # out = tf.reduce_max(aggall,axis=-1)
        out = tf.layers.conv1d(aggall, out_sz, 1, activation=activation)
        out = tf.layers.conv1d(out, out_sz, 1, activation=activation)
        out = tf.nn.l2_normalize(out,axis=-2)
        # out = tf.nn.softmax(out,axis=-1)
        return out, aggall


def attn_e2v_BNF_V2(seq, layer_id, out_sz, n_heads, net_mat, activation, in_drop=0.0, residual=False):
    with tf.name_scope('my_AE_BNF'):
        #seq : [batch_size, nb_nodes, features]
        #net_mat: [batch_size, nb_nodes, nb_nodes]
        [batch_size, nb_nodes, features] = seq.get_shape().as_list()
        # inpulse1 = tf.get_variable("inpulse1_ly"+str(layer_id),shape=[batch_size, nb_nodes],trainable=True)
        # impulse1 = tf.layers.conv1d(seq,nb_nodes,[nb_nodes])

        aggall_out = []

        thred_net_mat = tf.to_float(net_mat>0)

        seq_out=seq
        for rep in range(5):
            aggall1 = []
            for i in range(n_heads):
                att_agg = attn_v2e_BNF_V2(seq_out)
                stimulate_mag_1 = tf.Variable(tf.ones([1]),trainable=True)
                att_agg = tf.multiply(tf.ones([nb_nodes,nb_nodes]),stimulate_mag_1)+att_agg
                stimulate_mag_2 = tf.Variable(tf.ones([1]),trainable=True)
                net_mat1 = tf.multiply(thred_net_mat,stimulate_mag_2)+net_mat
                agg1 = tf.matmul(tf.multiply(net_mat1, att_agg),seq_out)
                agg1 = tf.layers.conv1d(agg1, out_sz, 1, activation=activation)
                aggall1.append(agg1)
            aggall1 = tf.concat(aggall1, axis=-1)
            seq_out1 = tf.layers.conv1d(aggall1, out_sz, 1, activation=activation)

            if residual:
                if seq_out.shape[-1] != seq_out1.shape[-1]:
                    seq_out = seq_out1+tf.layers.conv1d(seq_out,seq_out1.shape[-1],1)
                else:
                    seq_out += seq_out1
            else:
                seq_out = seq_out1

            aggall_out.append(seq_out)

        # aggall2 = []
        # for i in range(n_heads):
        #     att_agg = attn_v2e_BNF_V2(seq_out1)
        #     stimulate_mag_1 = tf.Variable(tf.ones([1]),trainable=True)
        #     att_agg = tf.multiply(tf.eye(nb_nodes,nb_nodes),stimulate_mag_1)+att_agg
        #     stimulate_mag_2 = tf.Variable(tf.ones([1]), trainable=True)
        #     net_mat1 = tf.multiply(thred_net_mat, stimulate_mag_2) + net_mat
        #     agg1 = tf.matmul(tf.multiply(net_mat1, att_agg), seq)
        #     agg1 = tf.layers.conv1d(agg1, out_sz, 1, activation=activation)
        #     aggall2.append(agg1)
        # aggall2 = tf.concat(aggall2, axis=-1)
        # seq_out2 = tf.layers.conv1d(aggall2, out_sz, 1, activation=activation)
        #
        # aggall3 = []
        # for i in range(n_heads):
        #     att_agg = attn_v2e_BNF_V2(seq_out2)
        #     stimulate_mag_1 = tf.Variable(tf.ones([1]), trainable=True)
        #     att_agg = tf.multiply(tf.eye(nb_nodes, nb_nodes), stimulate_mag_1) + att_agg
        #     stimulate_mag_2 = tf.Variable(tf.ones([1]), trainable=True)
        #     net_mat1 = tf.multiply(thred_net_mat, stimulate_mag_2) + net_mat
        #     agg1 = tf.matmul(tf.multiply(net_mat1, att_agg), seq)
        #     agg1 = tf.layers.conv1d(agg1, out_sz, 1, activation=activation)
        #     aggall3.append(agg1)
        # aggall3 = tf.concat(aggall3, axis=-1)
        # seq_out3 = tf.layers.conv1d(aggall3, out_sz, 1, activation=activation)

        aggall = tf.concat(aggall_out, axis=-1)

        out = tf.layers.conv1d(aggall, out_sz, 1, activation=activation)
        out = tf.layers.conv1d(out, out_sz, 1, activation=activation)
        out = tf.nn.l2_normalize(out,axis=-2)
        # out = tf.nn.softmax(out,axis=-1)
        return out, aggall

def attn_v2e_BNF_V2(seq, activation=tf.nn.softmax, in_drop=0.0, coef_drop=0.0, residual=False, n_heads=2, feat_dim=64):
    with tf.name_scope('my_AE_BNF'):
        seq = tf.layers.conv1d(seq,feat_dim,1,use_bias=False)
        f_1 = tf.layers.conv1d(seq, 1, 1)
        f_2 = tf.layers.conv1d(seq, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.leaky_relu(logits)
        # coefs = tf.contrib.layers.bias_add(activation(logits))
        coefs = activation(coefs)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

        # if residual:
        #     jump_net = []
        #     jump_net.append(tf.expand_dims(new_net,axis=-1))
        #     e2e_net = tf.expand_dims(new_net,axis=-1)
        #     e2e_out_size = 8
        #     for i in range(n_heads):
        #         e2e_net = attn_e2e_BNF(e2e_out_size, e2e_net, activation=tf.nn.tanh)*10
        #         jump_net.append(e2e_net)
        #     jump_net = tf.concat(jump_net,axis=-1)
        #     # _, _, _, filters = jump_net.get_shape()
        #     jump_net = tf.layers.conv2d(jump_net,1,1,activation=tf.nn.tanh)*10
        #     new_net = tf.squeeze(jump_net,axis=-1)
        #     # new_net = tf.reduce_max(jump_net,axis=-1)


        return coefs

def attn_v2e_BNF(seq, activation, in_drop=0.0, residual=False, n_heads=2):
    with tf.name_scope('my_AE_BNF'):
        seq = tf.layers.conv1d(seq,256,1,use_bias=False)
        f_1 = tf.layers.conv1d(seq, 1, 1)
        f_2 = tf.layers.conv1d(seq, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.tanh(logits)*10
        # coefs = tf.contrib.layers.bias_add(activation(logits))
        # coefs = tf.nn.softmax(coefs)
        new_net = (coefs + tf.transpose(coefs,perm=[0,2,1]))/2

        if residual:
            jump_net = []
            jump_net.append(tf.expand_dims(new_net,axis=-1))
            e2e_net = tf.expand_dims(new_net,axis=-1)
            e2e_out_size = 8
            for i in range(n_heads):
                e2e_net = attn_e2e_BNF(e2e_out_size, e2e_net, activation=tf.nn.tanh)*10
                jump_net.append(e2e_net)
            jump_net = tf.concat(jump_net,axis=-1)
            # _, _, _, filters = jump_net.get_shape()
            jump_net = tf.layers.conv2d(jump_net,1,1,activation=tf.nn.tanh)*10
            new_net = tf.squeeze(jump_net,axis=-1)
            # new_net = tf.reduce_max(jump_net,axis=-1)


        return new_net

def attn_e2e_BNF(out_sz, net_mat, activation, in_drop=0.0, residual=False):
    with tf.name_scope('my_AE_BNF'):
        _, n_nodes, _, filters = net_mat.get_shape()
        reshape_net = tf.reshape(net_mat, shape=[-1, n_nodes, filters, 1])
        conv1 = tf.layers.conv2d(reshape_net, out_sz, kernel_size=[n_nodes, filters])
        conv2 = tf.layers.conv2d(reshape_net, out_sz, kernel_size=[n_nodes, filters])
        re_conv1 = tf.reshape(conv1, shape=[-1, n_nodes, 1, out_sz])
        re_conv1 = tf.transpose(re_conv1, perm=[0, 3, 1, 2])
        re_conv2 = tf.reshape(conv2, shape=[-1, n_nodes, 1, out_sz])
        re_conv2 = tf.transpose(re_conv2, perm=[0, 3, 2, 1])
        # re_conv2 = tf.reshape(conv2, shape=[-1, 1, n_nodes])
        add1 = re_conv1 + re_conv2

        add1 = (add1+tf.transpose(add1,perm=[0,1,3,2]))/2

        out = tf.transpose(add1, perm=[0, 2, 3, 1])
        return activation(out)



def InnerProductDecoder(seq,activation=tf.nn.sigmoid,dropout=0.):
    inputs = tf.nn.dropout(seq, 1 - dropout)
    x = tf.transpose(inputs,perm=[0,2,1])
    x = tf.matmul(inputs, x)
    x = tf.reshape(x, [-1])
    outputs = activation(x)
    return outputs

