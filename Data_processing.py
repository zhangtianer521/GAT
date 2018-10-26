import sklearn as skl
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample
import csv
import pandas as pd
import nilearn.connectome as connectome

import pickle


def data_reorder(Datadir):
    ### read fmri signal data (.npy) and DTI network data (matlab matrix)
    with open(Datadir+'DTI_passed_subj.txt','r') as f:
        filenames = f.read().splitlines()

    # Datadir = '/home/wen/Documents/gcn_kifp/Data/'
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9 ]  # small number more sparse


    # fmri_nets = []
    fmri_signals = []

    DTI_connects = []
    for file in filenames:

        # DTI_connectivity = np.loadtxt(Datadir+file+'_fdt_matrix')
        DTI_connectivity = np.load(Datadir+file+'_DTInetworks.npy')

        ######################## need to fix, some subjects miss a brain region
        if DTI_connectivity.shape[0] == 246:
            DTI_connectivity[DTI_connectivity < 5] = 0
            row_sum = np.sum(DTI_connectivity,axis=1)
            row_sum[row_sum==0]=0.01
            # DTI_connectivity[DTI_connectivity > 0] = 1
            DTI_connectivity = DTI_connectivity/row_sum[:,np.newaxis]
            np.fill_diagonal(DTI_connectivity,1)
            # DTI_connectivity = DTI_connectivity / (DTI_connectivity.sum(axis=1).transpose())
            # DTI_connectivity = (DTI_connectivity.transpose() + DTI_connectivity) / 2
            # DTI_connectivity = sparsity(DTI_connectivity,sparsity_levels)
            DTI_connects.append(DTI_connectivity)

            fmri_signal = np.load(Datadir + file + '.npy')
            fmri_signals.append(fmri_signal)

            # estimator = connectome.ConnectivityMeasure()
            # fmri_net = estimator.fit_transform([fmri_signal])[0]
            # fmri_net = np.abs(fmri_net)
            # fmri_net = normalize_mat(fmri_net)
            # fmri_nets.append(fmri_net)


        ########################



    ### stack the data in the 3rd dimension

    # fmri_nets = np.stack(fmri_nets,axis=0)
    fmri_signals = np.stack(fmri_signals,axis=0)
    # fmri_nets = np.transpose(fmri_nets,(2,0,1))

    DTI_connects = np.stack(DTI_connects,axis=0)
    # DTI_connects = np.transpose(DTI_connects, (2, 0, 1, 3))


    return fmri_signals, DTI_connects

def load_data(Datadir, labelscsv, augmentation = 0):  ### labels: a cvs file
    fmri_signals, DTI_net = data_reorder(Datadir)


    graphs = np.zeros(DTI_net.shape)

    # for i in range(graphs.shape[3]):
    #     graphs[...,i] = np.multiply(fmri_net,DTI_net[...,i])

    # features = np.ones((graphs.shape[0],graphs.shape[1],1, graphs.shape[3]))

    #normalize features
    # for ind in range(features.shape[0]):
    #     features[ind,:,:]*=255/features[ind,:,:].max()


    # read cvs file [id, label]
    labels = []
    with open(labelscsv) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line = line +1
                continue
            labels.append(row[1])

    # generate training and testing samples

    train_fmri_signals, test_fmri_signals, train_labels, test_labels, train_graphs, test_graphs = train_test_split(fmri_signals,labels,graphs,test_size=0.2)
    train_fmri_signals, val_fmri_signals, train_labels, val_labels, train_graphs, val_graphs = train_test_split(train_fmri_signals, train_labels, train_graphs, test_size=0.2)
    # generate augmentation data

    train_fmri_net, train_labels , train_graph= augmentation_fmri_net(train_fmri_signals,train_labels,train_graphs,augmentation)
    val_fmri_net, val_labels, val_graph = augmentation_fmri_net(val_fmri_signals, val_labels, val_graphs, 0)
    test_fmri_net, test_labels , test_graph= augmentation_fmri_net(test_fmri_signals, test_labels, test_graphs, 0)


    # labels = np.asarray(labels)
    # indices = sample(range(features.shape[0]),int(features.shape[0]*0.8))
    # train_features = features[indices]
    # test_features = np.delete(features,indices)
    # train_labels = labels[indices]
    # test_labels = np.delete(labels,indices)


    ### features: 3D array, [#subjects, #nodes, #feature_per_node]
    ### graph: 2D array
    ### labels: 1D list

    return train_fmri_net, train_graph, train_labels, val_fmri_net, val_graph, val_labels, test_fmri_net, test_graph, test_labels


def sparsity(net, sparsity_level):
    net_vec = np.matrix.flatten(net)
    net_sort = np.sort(net_vec)
    sparsity_mat = []
    for sparsity in sparsity_level:
        sp_tmp = np.array(net)
        threshould = net_sort[int(len(net_sort)*(1-sparsity))]
        sp_tmp[sp_tmp<threshould]=0
        sp_tmp[sp_tmp>0]=1
        sparsity_mat.append(sp_tmp)
    return np.stack(sparsity_mat,axis=-1)


def signals_to_net(signal):
    estimator = connectome.ConnectivityMeasure()
    fmri_net = estimator.fit_transform([signal])[0]
    fmri_net = np.abs(fmri_net)
    fmri_net = normalize_mat(fmri_net)
    return fmri_net

def augmentation_fmri_net(signals, labels, graphs, augsize):
    aug_nets = []
    aug_labels =[]
    aug_graphs = []
    if augsize<=signals.shape[0]:
        for i in range(signals.shape[0]):
            aug_nets.append(signals_to_net(signals[i,...]))
        return np.stack(aug_nets,axis=0), labels, graphs
    else:
        for i in range(augsize):
            ind = np.random.randint(signals.shape[0],size=1)[0]
            endp = np.random.randint(int(signals.shape[1]/3-1),size=1)[0]  # signal in dim1
            aug_signal = signals[ind,endp:endp+int(signals.shape[1]/3*2),:]
            aug_nets.append(signals_to_net(aug_signal))
            aug_labels.append(labels[ind])
            aug_graphs.append(graphs[ind:ind+1,...])
        aug_labels = np.stack(aug_labels, axis=0)
        aug_graphs = np.stack(aug_graphs, axis=0)
    return np.stack(aug_nets,axis=0), aug_labels, np.squeeze(aug_graphs,axis=1)



def one_hot(labels):
    s = pd.Series(labels)
    return pd.get_dummies(s)

def normalize_mat(net):
    flatten = np.matrix.flatten(net)
    sort = np.sort(flatten)
    net = net/sort[-1]*1
    return net


def creat_csv_Autism(idtxt, allcsvlist,outcsv):
    with open(idtxt,'r') as f:
        filenames = f.read().splitlines()

    with open(outcsv,'w') as outf:
        spamwriter = csv.writer(outf,delimiter=',')
        spamwriter.writerow(['id','labels'])
        for file in filenames:
            for csv_s in allcsvlist:
                with open(csv_s) as f_csv:
                    csv_reader = csv.reader(f_csv,delimiter = ',')
                    line = 0
                    for row in csv_reader:
                        if line ==0:
                            line +=1
                            continue
                        if file in row:
                            spamwriter.writerow([file ,row[3]])

def creat_csv_Schiz(idtxt, allcsvlist,outcsv):
    with open(idtxt,'r') as f:
        filenames = f.read().splitlines()

    with open(outcsv,'w') as outf:
        spamwriter = csv.writer(outf,delimiter=',')
        spamwriter.writerow(['id','labels'])
        for file in filenames:
            for csv_s in allcsvlist:
                with open(csv_s) as f_csv:
                    csv_reader = csv.reader(f_csv,delimiter = '\t')
                    line = 0
                    for row in csv_reader:
                        if line ==0:
                            line +=1
                            continue
                        if file in row:
                            if 'No_Known_Disorder' in row[4]: label = 1
                            else: label = 2
                            spamwriter.writerow([file ,label])


def load_sorted_data(dir):
    pkl_file = open(dir, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()

    train_data = data['train']
    train_fmri_net = train_data['fmri_net']
    train_graphs = train_data['graph']
    train_labels = train_data['label']

    val_data = data['val']
    val_fmri_net = val_data['fmri_net']
    val_graphs = val_data['graph']
    val_labels = val_data['label']

    test_data = data['test']
    test_fmri_net = test_data['fmri_net']
    test_graphs = test_data['graph']
    test_labels = test_data['label']

    return train_fmri_net, train_graphs, train_labels, val_fmri_net, val_graphs, val_labels, test_fmri_net, test_graphs, test_labels


if __name__ == '__main__':
    # data_reorder('/home/wen/Documents/gcn_kifp/Data/')
    # load_data('../Data/', '../Data/labels.csv')

    # # Autism
    # csv_dir = '/mnt/wzhan139/Image_Data/Network_data/Autism/'
    # csvfiles = [csv_dir + 'ABIDEII-BNI_1.csv',
    #             csv_dir + 'ABIDEII-IP_1.csv',
    #             csv_dir + 'ABIDEII-NYU_1.csv',
    #             csv_dir + 'ABIDEII-NYU_2.csv',
    #             csv_dir + 'ABIDEII-SDSU_1.csv',
    #             csv_dir + 'ABIDEII-TCD_1.csv']
    # creat_csv_Autism('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Autism/DTI_passed_subj.txt',
    #           csvfiles, '/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Autism/labels.csv')


    # # Schiz
    # csvfiles = ['/mnt/easystore_8T/Wen_Data/Schizophrenia/COBRE/participants.tsv']
    # creat_csv_Schiz('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Schiz/DTI_passed_subj.txt',
    #           csvfiles, '/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Schiz/labels.csv')

    train_fmri_net, train_adj, train_labels, val_fmri_net, val_adj, val_labels, test_fmri_net, test_adj, test_labels = \
        load_data('Data_BNF/Schiz/', 'Data_BNF/Schiz/labels.csv', 500)

    train_data = {}
    train_data['fmri_net'] = train_fmri_net
    train_data['graph'] = train_adj
    train_data['label'] = train_labels

    val_data = {}
    val_data['fmri_net'] = val_fmri_net
    val_data['graph'] = val_adj
    val_data['label'] = val_labels

    test_data = {}
    test_data['fmri_net'] = test_fmri_net
    test_data['graph'] = test_adj
    test_data['label'] = test_labels

    data = {}
    data['train'] = train_data
    data['test'] = test_data
    data['val'] =val_data

    pkl_file = open('Data_BNF/Schiz/sort_data.pkl', 'wb')
    pickle.dump(data,pkl_file)
    pkl_file.close()

