import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits


class GAT_BNF(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, nb_nodes, nb_slots, training, attn_drop, ffd_drop,
                  reweight_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head_BNF(inputs, reweight_mat=reweight_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-2) # last dimension is nb_slot not nb_filters
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head_BNF(h_1, reweight_mat=reweight_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-2)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head_BNF(h_1, reweight_mat=reweight_mat,
                                        out_sz=5, activation=activation,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        Graph_out = tf.add_n(out) / n_heads[-1]  # Graph_out: [batch_size, nb_nodes, 1, nb_slots]
        # Graph_out = tf.squeeze(Graph_out,axis=2)
        #
        # Slot_out = tf.layers.conv1d(Graph_out,1,1,use_bias=True)
        # Slot_out = tf.squeeze(Slot_out,axis=2)  # Slot_out: [batch_size, nb_nodes]
        # # Slot_out = tf.contrib.layers.bias_add(Slot_out)
        # node_weight = tf.nn.softmax(Slot_out)
        # Slot_out = tf.nn.relu(Slot_out)


        # Graph_out = tf.reshape(Graph_out,shape=[-1,nb_nodes*nb_slots])
        Graph_out = tf.layers.flatten(Graph_out)
        logits = tf.layers.dense(Graph_out,nb_classes)
        # logits = tf.contrib.layers.bias_add(node_out)


        return logits, tf.nn.softmax(logits)
