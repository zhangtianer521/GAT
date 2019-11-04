import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

# class GAT(BaseGAttN):
#     def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
#             bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
#         attns = []
#         for _ in range(n_heads[0]):
#             attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
#                 out_sz=hid_units[0], activation=activation,
#                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
#         h_1 = tf.concat(attns, axis=-1)
#         for i in range(1, len(hid_units)):
#             h_old = h_1
#             attns = []
#             for _ in range(n_heads[i]):
#                 attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
#                     out_sz=hid_units[i], activation=activation,
#                     in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
#             h_1 = tf.concat(attns, axis=-1)
#         out = []
#         for i in range(n_heads[-1]):
#             out.append(layers.attn_head(h_1, bias_mat=bias_mat,
#                 out_sz=nb_classes, activation=lambda x: x,
#                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
#         logits = tf.add_n(out) / n_heads[-1]
#
#         return logits


class GAT_BNF(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_slots, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # sparsity net to get reweight_mat
        sparsity_mat = layers.sparsity_BNF(net_mat,nb_slots)
        fmri_mat = tf.expand_dims(fmri_net,3)
        reweight_mat = tf.multiply(sparsity_mat,fmri_mat)
        # reweight_mat = tf.nn.l2_normalize(reweight_mat,axis=2)
        # reweight_mat = reweight_mat/tf.reduce_sum(sparsity_mat,axis=2,keep_dims=True)

        # reweight_mat = tf.contrib.layers.bias_add(reweight_mat)


        jump_out = []
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head_BNF(inputs, reweight_mat=reweight_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          residual=False))
        h_1 = tf.concat(attns, axis=-2) # last dimension is nb_slot not nb_filters
        jump_out.append(h_1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head_BNF(h_1, reweight_mat=reweight_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              residual=residual))

            h_1 = tf.concat(attns, axis=-2)
            jump_out.append(h_1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head_BNF(h_1, reweight_mat=reweight_mat,
                                        out_sz=5, activation=lambda x: x,
                                        residual=False))
        Graph_out = tf.add_n(out) / n_heads[-1]  # Graph_out: [batch_size, nb_nodes, nb_filters, nb_slots]
        Graph_out = activation(Graph_out)
        jump_out.append(Graph_out)
        jump_out = tf.concat(jump_out,axis=-2)
        jump_out = tf.reduce_max(jump_out,axis=2)

        recon_net = tf.matmul(jump_out,tf.transpose(jump_out,perm=[0,2,1]))


        # _, _, nb_filters,nb_slots=jump_out.get_shape()
        # jump_out = tf.reshape(jump_out,shape=[-1,nb_nodes,nb_filters*nb_slots])
        # fc1 = tf.layers.conv1d(jump_out,50,1,activation=activation)
        fc2 = tf.layers.conv1d(jump_out, 1, 1, activation=activation)

        # fc1 = tf.squeeze(fc1,axis=-1)
        #
        # Slot_out = tf.layers.conv1d(Graph_out,1,1,use_bias=True)
        # Slot_out = tf.squeeze(Slot_out,axis=2)  # Slot_out: [batch_size, nb_nodes]
        # # Slot_out = tf.contrib.layers.bias_add(Slot_out)
        # node_weight = tf.nn.softmax(Slot_out)
        # Slot_out = tf.nn.relu(Slot_out)


        # Graph_out = tf.reshape(Graph_out,shape=[-1,nb_nodes*nb_slots])
        Graph_out = tf.layers.flatten(fc2)
        logits = tf.layers.dense(Graph_out,nb_classes)
        # logits = tf.contrib.layers.bias_add(node_out)



        return logits, tf.nn.sigmoid(recon_net)

class GAT_AE_BNF(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_CAM, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.relu, residual=False):


        def reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu):
            # block 1
            layer_id = 0
            node_emb = []
            edge_emb =[]
            net_mat = tf.contrib.layers.batch_norm(net_mat)
            e2v1_init, agg1 = layers.attn_e2v_BNF(inputs, layer_id, out_sz=hid_units[0], n_heads=n_heads[0], net_mat=net_mat, activation=activation)
            v2e1_init1 = layers.attn_v2e_BNF(e2v1_init,activation,residual=True)
            edge_emb.append(v2e1_init1)
            v2e1=v2e1_init1
            # v2e1_init2 = layers.attn_v2e_BNF(e2v1_init, activation,residual=True)
            # v2e1_init3 = layers.attn_v2e_BNF(e2v1_init, activation,residual=True)
            # v2e1 = tf.add_n([v2e1_init1,v2e1_init2,v2e1_init3])/3
            node_emb.append(e2v1_init)
            layer_id +=1

            #other blocks
            for i in range(len(hid_units)-1):
                # v2e1 = tf.contrib.layers.batch_norm(v2e1)
                e2v1, _ = layers.attn_e2v_BNF(inputs, layer_id, net_mat=v2e1, n_heads=n_heads[i], out_sz=hid_units[i], activation=activation)
                v2e1_tmp = []
                for rep in range(10):
                    v2e1 = layers.attn_v2e_BNF(e2v1, activation)
                    v2e1_tmp.append(v2e1)
                # v2e1_2 = layers.attn_v2e_BNF(e2v1, activation)
                # v2e1_3 = layers.attn_v2e_BNF(e2v1, activation)
                v2e1 = tf.add_n(v2e1_tmp) / 10
                node_emb.append(e2v1)
                edge_emb.append(v2e1)
                layer_id += 1

            #final blocks
            # v2e1 = tf.contrib.layers.batch_norm(v2e1)
            e2v1, _ = layers.attn_e2v_BNF(inputs, layer_id, net_mat=v2e1, n_heads=n_heads[-1], out_sz=hid_units[-1], activation=activation)
            node_emb.append(e2v1)
            e2v1_final = tf.concat(node_emb,axis=-1)
            # if training==True:
            #     e2v1_final = tf.nn.dropout(e2v1_final,0.7)
            graph_final = layers.attn_v2e_BNF(e2v1_final,activation,residual=True)
            edge_emb.append(graph_final)
            edge_emb=tf.stack(edge_emb,axis=-1)
            graph_final=tf.layers.conv2d(edge_emb,1,1,activation=tf.nn.tanh)
            graph_final=tf.squeeze(graph_final,axis=-1)
            if training==True:
                e2v1_final = tf.nn.dropout(e2v1_final,0.7)

            node_fc1 = tf.layers.conv1d(e2v1_final,30,1,activation=activation)

            return node_fc1, graph_final

        _, nb_nodes, _ = net_mat.get_shape()
        node_fc1_DTI, reconstruct_fmri = reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu)
        node_fc1_fmri, reconstruct_DTI = reconstruction(inputs, training, fmri_net, hid_units, n_heads, activation=tf.nn.relu)
        node_fc1 = tf.concat([node_fc1_fmri,node_fc1_DTI],axis=-1)
        # node_fc2 = tf.layers.conv1d(node_fc1, 32, 1, activation=activation)

        # node_fc2 = tf.layers.flatten(node_fc2)
        # if training==True:
        #     node_fc2 = tf.nn.dropout(node_fc2,0.5)
        # node_fc2 = tf.layers.dense(node_fc2, 256,activation=activation)
        # if training==True:
        #     node_fc2 = tf.nn.dropout(node_fc2,0.5)
        # node_fc2 = tf.layers.dense(node_fc2, 30,activation=activation)
        # digits = tf.layers.dense(node_fc2, nb_classes, name='dense_out')

        node_fc2 = tf.layers.conv1d(node_fc1, nb_CAM, 1, activation=activation)
        # node_fc2 = tf.layers.conv1d(node_fc2, nb_CAM, 1, activation=activation)
        node_fc3 = tf.reduce_mean(node_fc2,axis=-2)
        digits = tf.layers.dense(node_fc3, nb_classes, name='dense_out')

        with tf.variable_scope('dense_out',reuse=True):
            w = tf.get_variable('kernel')
        cam = tf.matmul(tf.reshape(node_fc2,shape=[-1,nb_CAM]),w)
        cam = tf.reshape(cam,shape=[-1,nb_nodes,nb_classes])



        return digits, reconstruct_fmri, reconstruct_DTI, tf.nn.softmax(digits), cam


class GAT_AE_BNF_V2(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_CAM, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.relu, residual=False):


        def reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu):
            # block 1
            layer_id = 0
            _, n_node, _ = inputs.get_shape()
            node_emb = []
            edge_emb =[]
            net_mat = tf.contrib.layers.batch_norm(net_mat)
            e2v1_init, agg1 = layers.attn_e2v_BNF_V2(inputs, layer_id, out_sz=hid_units[0], n_heads=n_heads[0], net_mat=net_mat, activation=activation, residual=True)

            graph_final = layers.InnerProductDecoder(e2v1_init)

            graph_final = tf.reshape(graph_final,[-1, n_node,n_node])
            # graph_final=tf.squeeze(graph_final,axis=-1)

            if training==True:
                e2v1_init = tf.nn.dropout(e2v1_init,0.7)

            node_fc1 = tf.layers.conv1d(e2v1_init,64,1,activation=activation)

            return node_fc1, graph_final

        _, nb_nodes, _ = net_mat.get_shape()
        node_fc1_fmri, reconstruct_fmri = reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu)
        node_fc1_DTI, reconstruct_DTI = reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu)
        node_fc1 = tf.concat([node_fc1_fmri,node_fc1_DTI],axis=-1)
        # node_fc2 = tf.layers.conv1d(node_fc1, 32, 1, activation=activation)

        # node_fc2 = tf.layers.flatten(node_fc2)
        # if training==True:
        #     node_fc2 = tf.nn.dropout(node_fc2,0.5)
        # node_fc2 = tf.layers.dense(node_fc2, 256,activation=activation)
        # if training==True:
        #     node_fc2 = tf.nn.dropout(node_fc2,0.5)
        # node_fc2 = tf.layers.dense(node_fc2, 30,activation=activation)
        # digits = tf.layers.dense(node_fc2, nb_classes, name='dense_out')

        node_fc2 = tf.layers.conv1d(node_fc1, nb_CAM, 1, activation=activation)
        # node_fc2 = tf.layers.conv1d(node_fc2, nb_CAM, 1, activation=activation)
        node_fc3 = tf.reduce_mean(node_fc2,axis=-2)
        digits = tf.layers.dense(node_fc3, nb_classes, name='dense_out')

        with tf.variable_scope('dense_out',reuse=True):
            w = tf.get_variable('kernel')
        cam = tf.matmul(tf.reshape(node_fc2,shape=[-1,nb_CAM]),w)
        cam = tf.reshape(cam,shape=[-1,nb_nodes,nb_classes])



        return digits, reconstruct_fmri, reconstruct_DTI, tf.nn.softmax(digits), cam



class GAT_AE_BNF_single(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_CAM, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.relu, residual=False):


        def reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu):
            # block 1
            layer_id = 0
            node_emb = []
            net_mat = tf.contrib.layers.batch_norm(net_mat)
            e2v1_init, agg1 = layers.attn_e2v_BNF(inputs, layer_id, out_sz=hid_units[0], n_heads=n_heads[0], net_mat=net_mat, activation=activation)
            v2e1_init1 = layers.attn_v2e_BNF(e2v1_init,activation,residual=True)
            v2e1=v2e1_init1
            # v2e1_init2 = layers.attn_v2e_BNF(e2v1_init, activation,residual=True)
            # v2e1_init3 = layers.attn_v2e_BNF(e2v1_init, activation,residual=True)
            # v2e1 = tf.add_n([v2e1_init1,v2e1_init2,v2e1_init3])/3
            node_emb.append(e2v1_init)
            layer_id +=1

            #other blocks
            for i in range(len(hid_units)-1):
                # v2e1 = tf.contrib.layers.batch_norm(v2e1)
                e2v1, _ = layers.attn_e2v_BNF(inputs, layer_id, net_mat=v2e1, n_heads=n_heads[i], out_sz=hid_units[i], activation=activation)
                v2e1_tmp = []
                for rep in range(10):
                    v2e1 = layers.attn_v2e_BNF(e2v1, activation)
                    v2e1_tmp.append(v2e1)
                # v2e1_2 = layers.attn_v2e_BNF(e2v1, activation)
                # v2e1_3 = layers.attn_v2e_BNF(e2v1, activation)
                v2e1 = tf.add_n(v2e1_tmp) / 10
                node_emb.append(e2v1)
                layer_id += 1

            #final blocks
            # v2e1 = tf.contrib.layers.batch_norm(v2e1)
            e2v1, _ = layers.attn_e2v_BNF(inputs, layer_id, net_mat=v2e1, n_heads=n_heads[-1], out_sz=hid_units[-1], activation=activation)
            node_emb.append(e2v1)
            e2v1_final = tf.concat(node_emb,axis=-1)
            if training==True:
                e2v1_final = tf.nn.dropout(e2v1_final,0.5)
            graph_final = layers.attn_v2e_BNF(e2v1_final,activation,residual=False)
            # if training==True:
            #     e2v1_final = tf.nn.dropout(e2v1_final,0.7)

            node_fc1 = tf.layers.conv1d(e2v1_final,30,1,activation=activation)

            return node_fc1, graph_final

        _, nb_nodes, _ = net_mat.get_shape()


        node_fc1_DTI, reconstruct_fmri = reconstruction(inputs, training, net_mat, hid_units, n_heads, activation=tf.nn.relu)
        # node_fc1_fmri, reconstruct_DTI = reconstruction(inputs, training, fmri_net, hid_units, n_heads, activation=tf.nn.relu)
        # node_fc1 = tf.concat([node_fc1_fmri,node_fc1_DTI],axis=-1)
        node_fc2 = tf.layers.conv1d(node_fc1_DTI, 1, 1, activation=activation)
        node_fc2 = tf.layers.flatten(node_fc2)
        if training==True:
            node_fc2 = tf.nn.dropout(node_fc2,0.7)
        node_fc2 = tf.layers.dense(node_fc2, 256,activation=activation)
        if training==True:
            node_fc2 = tf.nn.dropout(node_fc2,0.7)
        node_fc2 = tf.layers.dense(node_fc2, 30,activation=activation)
        digits = tf.layers.dense(node_fc2, nb_classes, name='dense_out')

        # node_fc2 = tf.layers.conv1d(node_fc1, nb_CAM, 1, activation=activation)
        # node_fc2 = tf.layers.conv1d(node_fc2, nb_CAM, 1, activation=activation)
        # node_fc3 = tf.reduce_mean(node_fc2,axis=-2)
        # digits = tf.layers.dense(node_fc3, nb_classes, name='dense_out')

        # with tf.variable_scope('dense_out',reuse=True):
        #     w = tf.get_variable('kernel')
        # cam = tf.matmul(tf.reshape(node_fc2,shape=[-1,nb_CAM]),w)
        # cam = tf.reshape(cam,shape=[-1,nb_nodes,nb_classes])



        return digits, reconstruct_fmri, reconstruct_fmri, tf.nn.softmax(digits)


class GAT_FC_BNF(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_CAM, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.relu, residual=False):

        fmri_net_flatten = tf.layers.flatten(fmri_net)
        net_mat_flatten = tf.layers.flatten(net_mat)
        net_flatten = tf.concat([fmri_net_flatten,net_mat_flatten],axis=-1)
        # if training==True:
        #     net_flatten= tf.nn.dropout(net_flatten,0.5)
        fc1 = tf.layers.dense(net_flatten, 64, activation=activation)
        if training==True:
            fc1= tf.nn.dropout(fc1,0.5)

        # fc2 = tf.layers.dense(fc1, 30, activation=activation)
        digits = tf.layers.dense(fc1, nb_classes)

class GAT_Node2Vec_BNF(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_CAM, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.relu, residual=False):
        # fmri_net_flatten = tf.layers.flatten(fmri_net)
        # net_mat_flatten = tf.layers.flatten(net_mat)
        # net_flatten = tf.concat([fmri_net_flatten, net_mat_flatten], axis=-1)
        # # if training==True:
        # #     net_flatten= tf.nn.dropout(net_flatten,0.5)
        # fc1 = tf.layers.dense(net_flatten, 64, activation=activation)
        # if training == True:
        #     fc1 = tf.nn.dropout(fc1, 0.5)
        #
        # # fc2 = tf.layers.dense(fc1, 30, activation=activation)
        # digits = tf.layers.dense(fc1, nb_classes)

        node_fc2 = tf.layers.conv1d(fmri_net, 256, 1, activation=activation)
        node_fc2 = tf.layers.conv1d(node_fc2, nb_CAM, 1, activation=activation)
        # node_fc2 = tf.layers.conv1d(node_fc2, nb_CAM, 1, activation=activation)
        node_fc3 = tf.reduce_mean(node_fc2, axis=-2)
        digits = tf.layers.dense(node_fc3, nb_classes, name='dense_out')




        return digits, tf.nn.softmax(digits)




class GAT_bi_BNF(BaseGAttN):
    # inputs dim: [batch_size, nb_nodes, nb_features, nb_slots]
    def inference(inputs, nb_classes, fmri_net, nb_slots, training,
                  net_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # sparsity net to get reweight_mat
        sparsity_DTI_mat = layers.sparsity_BNF(net_mat, nb_slots-1)
        sparsity_fmri_mat = layers.sparsity_BNF(fmri_net, nb_slots-1)

        fmri_mat = tf.expand_dims(fmri_net, 3)
        reweight_fmri_mat = tf.multiply(sparsity_DTI_mat, fmri_mat)
        reweight_fmri_mat = tf.concat([reweight_fmri_mat,fmri_mat],axis=-1)


        DTI_mat = tf.expand_dims(net_mat, 3)
        reweight_DTI_mat = tf.multiply(sparsity_fmri_mat, DTI_mat)
        reweight_DTI_mat = tf.concat([reweight_DTI_mat, DTI_mat], axis=-1)

        jump_fmri_out = []
        jump_DTI_out = []
        fmri_1 = layers.attn_head_BNF(inputs, reweight_mat=reweight_fmri_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              residual=False)
        DTI_1 = layers.attn_head_BNF(inputs, reweight_mat=reweight_DTI_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              residual=False)
        jump_fmri_out.append(fmri_1)
        jump_DTI_out.append(DTI_1)
        for i in range(1, len(hid_units)):
            fmri_1 = layers.attn_head_BNF(fmri_1, reweight_mat=reweight_fmri_mat,
                                                  out_sz=hid_units[i], activation=activation,
                                                  residual=residual)

            DTI_1 = layers.attn_head_BNF(DTI_1, reweight_mat=reweight_DTI_mat,
                                          out_sz=hid_units[i], activation=activation,
                                          residual=residual)
            jump_fmri_out.append(fmri_1)
            jump_DTI_out.append(DTI_1)

        jump_fmri_out = tf.concat(jump_fmri_out, axis=-2)
        jump_fmri_out = tf.reduce_max(jump_fmri_out, axis=2)

        jump_DTI_out = tf.concat(jump_DTI_out, axis=-2)
        jump_DTI_out = tf.reduce_max(jump_DTI_out, axis=2)

        recon_net_fmri = tf.matmul(jump_fmri_out, tf.transpose(jump_fmri_out, perm=[0, 2, 1]))
        recon_net_DTI = tf.matmul(jump_DTI_out, tf.transpose(jump_DTI_out, perm=[0, 2, 1]))

        # _, _, nb_filters,nb_slots=jump_out.get_shape()
        # jump_out = tf.reshape(jump_out,shape=[-1,nb_nodes,nb_filters*nb_slots])
        # fc1 = tf.layers.conv1d(jump_out,50,1,activation=activation)
        fc_fmri = tf.layers.conv1d(jump_fmri_out, 1, 1, activation=activation)
        fc_DTI = tf.layers.conv1d(jump_DTI_out, 1, 1, activation=activation)

        fc = tf.concat([fc_fmri,fc_DTI],axis=-1)

        # fc1 = tf.squeeze(fc1,axis=-1)
        #
        # Slot_out = tf.layers.conv1d(Graph_out,1,1,use_bias=True)
        # Slot_out = tf.squeeze(Slot_out,axis=2)  # Slot_out: [batch_size, nb_nodes]
        # # Slot_out = tf.contrib.layers.bias_add(Slot_out)
        # node_weight = tf.nn.softmax(Slot_out)
        # Slot_out = tf.nn.relu(Slot_out)

        # Graph_out = tf.reshape(Graph_out,shape=[-1,nb_nodes*nb_slots])
        Graph_out = tf.layers.flatten(fc)
        logits = tf.layers.dense(Graph_out, nb_classes)
        # logits = tf.contrib.layers.bias_add(node_out)

        return logits, tf.nn.sigmoid(recon_net_DTI),tf.nn.sigmoid(recon_net_fmri)


class Brainnetcnn(BaseGAttN):

    def inference(nb_classes, fmri_net, training,
                  net_mat, hid_units, dropout=0, activation=tf.nn.elu, residual=False):
        ### input: [batch_size, n_nodes, n_nodes]

        network1 = Brainnetcnn.network_structure(fmri_net,training, activation)
        network2 = Brainnetcnn.network_structure(net_mat,training, activation)

        concat = tf.concat([network1,network2],axis=-1)
        logit = tf.layers.dense(concat, nb_classes)
        return logit, network1, tf.nn.softmax(logit)


    def network_structure(net, training, activation=tf.nn.elu ):

        hide_s = [32, 32, 64, 256, 128, 30]
        batch_size, n_nodes, _ = net.get_shape()

        signal = tf.reshape(net, shape=[-1, n_nodes, 1, 1])
        conv1 = tf.layers.conv2d(signal, hide_s[0], kernel_size=[n_nodes, 1])
        conv2 = tf.layers.conv2d(signal, hide_s[0], kernel_size=[n_nodes, 1])
        re_conv1 = tf.reshape(conv1, shape=[-1, n_nodes, 1, hide_s[0]])
        re_conv1 = tf.transpose(re_conv1,perm=[0,3,1,2])
        re_conv2 = tf.reshape(conv2, shape=[-1, n_nodes, 1, hide_s[0]])
        re_conv2 = tf.transpose(re_conv2, perm=[0, 3, 2, 1])
        # re_conv2 = tf.reshape(conv2, shape=[-1, 1, n_nodes])
        add1 = re_conv1 + re_conv2
        add1 = activation(add1)

        add1 = tf.transpose(add1, perm=[0,2,3,1])

        add1 = tf.reshape(add1, shape=[-1, n_nodes, hide_s[0], 1])
        conv1 = tf.layers.conv2d(add1, hide_s[1], kernel_size=[n_nodes, hide_s[0]])
        conv2 = tf.layers.conv2d(add1, hide_s[1], kernel_size=[n_nodes, hide_s[0]])
        re_conv1 = tf.reshape(conv1, shape=[-1, n_nodes, 1, hide_s[1]])
        re_conv1 = tf.transpose(re_conv1,perm=[0,3,1,2])
        re_conv2 = tf.reshape(conv2, shape=[-1, n_nodes, 1, hide_s[1]])
        re_conv2 = tf.transpose(re_conv2, perm=[0, 3, 2, 1])
        add2 = re_conv1 + re_conv2
        add2 = activation(add2)

        add2 = tf.transpose(add2, perm=[0, 2, 3, 1])

        # e2n
        add2 = tf.reshape(add2, shape=[-1, n_nodes, hide_s[1], 1])
        conv1 = tf.layers.conv2d(add2, hide_s[2], kernel_size=[n_nodes, hide_s[1]], activation=activation)

        # n2g
        add3 = tf.reshape(conv1, shape=[-1, n_nodes, hide_s[2], 1])
        conv1 = tf.layers.conv2d(add3, hide_s[3], kernel_size=[n_nodes, hide_s[2]], activation=activation)
        fc1 = tf.reshape(conv1, shape=[-1, hide_s[3]])

        # fc
        fc2 = tf.layers.dense(fc1, hide_s[4], activation=activation)
        if training == True:
            fc2 = tf.nn.dropout(fc2, 0.5)
        fc3 = tf.layers.dense(fc2, hide_s[5], activation=activation)
        if training == True:
            fc3 = tf.nn.dropout(fc3, 0.5)
        # fc4 = tf.layers.dense(fc3, 2, activation=activation)

        return fc3

    def BNCNN_loss(logits, labels):
        # sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        # xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits), sample_wts)
        # return tf.reduce_mean(xentropy, name='xentropy_mean')

        classify_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # Recons_loss = tf.reduce_sum(Recons_loss,axis=1)
        return tf.reduce_mean(classify_loss)



    def BNCNN_syn_loss(logits, labels):
        # sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        # xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits), sample_wts)
        # return tf.reduce_mean(xentropy, name='xentropy_mean')

        classify_loss = tf.losses.mean_squared_error(labels, logits)
        # Recons_loss = tf.reduce_sum(Recons_loss,axis=1)
        return classify_loss


    def BNCNN_training(loss, lr):
        # weight decay
        vars = tf.trainable_variables()

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss)

        return train_op

    def BNCNN_accuracy(preds, labels):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

    def BNCNN_syn_accuracy(preds, labels):
        preds = tf.squeeze(preds)
        labels = tf.squeeze(labels)
        preds_mean = tf.reduce_mean(preds)
        labels_mean = tf.reduce_mean(labels)
        nominator = tf.reduce_sum((preds - preds_mean) * (labels - labels_mean))
        denominator = tf.sqrt(
            tf.reduce_sum(tf.square(preds - preds_mean)) * tf.reduce_sum(tf.square(labels - labels_mean)))
        corref = nominator / (denominator + tf.keras.backend.epsilon())
        return corref




