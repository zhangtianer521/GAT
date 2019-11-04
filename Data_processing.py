import sklearn as skl
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample
import csv
import pandas as pd
import nilearn.connectome as connectome
import random
import pickle
import os
import sys
import shutil
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, roc_auc_score
from collections import Counter
import matplotlib.pyplot as plt


def data_reorder(Datadir):
    ### read fmri signal data (.npy) and DTI network data (matlab matrix)
    with open(Datadir+'DTI_passed_subj.txt','r') as f:
        filenames = f.read().splitlines()


    # Datadir = '/home/wen/Documents/gcn_kifp/Data/'
    # sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9 ]  # small number more sparse


    # fmri_nets = []
    fmri_signals = []
    ROI_all = []

    DTI_connects = []
    for file in filenames:

        # DTI_connectivity = np.loadtxt(Datadir+file+'_fdt_matrix')
        DTI_connectivity = np.load(Datadir+file+'_DTInetworks.npy')

        if not os.path.isfile(Datadir+file+'_ROI.txt'):
            ROI_size = np.ones(DTI_connectivity.shape[0])
        else:
            ROI_size = read_roi_size(Datadir+file+'_ROI.txt')

        ROI_all.append(ROI_size)

        ######################## need to fix, some subjects miss a brain region
        if DTI_connectivity.shape[0] == 246:
            DTI_connectivity[DTI_connectivity < 10] = 0
            # row_sum = np.sum(DTI_connectivity,axis=1)
            # row_sum[row_sum==0]=0.0000001
            # ROI_size[ROI_size==0] = 0.0001
            # ROI_size = np.reciprocal(ROI_size)
            DTI_connectivity_norm = (DTI_connectivity.T / ROI_size/5000).T
            # np.fill_diagonal(DTI_connectivity_norm,5000)
            # DTI_connectivity = DTI_connectivity / (DTI_connectivity.sum(axis=1).transpose())
            DTI_connectivity_norm = (DTI_connectivity_norm.transpose() + DTI_connectivity_norm) / 2
            # DTI_connectivity[DTI_connectivity > 1e-3] = 1
            # DTI_connectivity[DTI_connectivity <1] = 0
            # DTI_connectivity = sparsity(DTI_connectivity,sparsity_levels)
            DTI_connects.append(DTI_connectivity_norm)

            fmri_signal = np.load(Datadir + file + '.npy')
            fmri_signals.append(fmri_signal)

            # estimator = connectome.ConnectivityMeasure()
            # fmri_net = estimator.fit_transform([fmri_signal])[0]
            # fmri_net = np.abs(fmri_net)
            # fmri_net = normalize_mat(fmri_net)
            # fmri_nets.append(fmri_net)


        ########################



    ### stack the data in the 3rd dimension
    ROI_all = np.stack(ROI_all,axis=1)
    # fmri_nets = np.stack(fmri_nets,axis=0)
    fmri_signals = np.stack(fmri_signals,axis=0)
    # fmri_nets = np.transpose(fmri_nets,(2,0,1))

    DTI_connects = np.stack(DTI_connects,axis=0)
    # DTI_connects = np.transpose(DTI_connects, (2, 0, 1, 3))


    return fmri_signals, DTI_connects


def net2edgelist(Datadir, outdir, featdir):
    with open(Datadir+'DTI_passed_subj.txt','r') as f:
        filenames = f.read().splitlines()


    # Datadir = '/home/wen/Documents/gcn_kifp/Data/'
    # sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9 ]  # small number more sparse
    sys.path.append('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/node2vec-master')

    for file in filenames:

        # DTI_connectivity = np.loadtxt(Datadir+file+'_fdt_matrix')
        DTI_connectivity = np.load(Datadir+file+'_DTInetworks.npy')

        if not os.path.isfile(Datadir+file+'_ROI.txt'):
            ROI_size = np.ones(DTI_connectivity.shape[0])
        else:
            ROI_size = read_roi_size(Datadir+file+'_ROI.txt')


        ######################## need to fix, some subjects miss a brain region
        if DTI_connectivity.shape[0] == 246:
            DTI_connectivity[DTI_connectivity < 10] = 0
            DTI_connectivity_norm = (DTI_connectivity.T / ROI_size/5000).T
            DTI_connectivity_norm = (DTI_connectivity_norm.transpose() + DTI_connectivity_norm) / 2

            DTI_connectivity_thred = (DTI_connectivity_norm>0).astype(int)

            if not os.path.isfile(outdir+file+'.edgelist'):
                with open(outdir+file+'.edgelist','w') as f:
                    for i in range(246):
                        for j in range(i,246):
                            if DTI_connectivity_thred[i,j]>0 and i!=j:
                                f.write(str(i)+' '+str(j)+'\n')

            os.chdir('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/node2vec-master')
            cmd = 'python src/main.py --input graph/HCP/'+file+'.edgelist --output emb/HCP/'+file+'.emb'
            os.system(cmd)





def read_roi_size(Datadir):
    with open(Datadir,'r') as f:
        roi_size = np.ones([246],dtype=int)
        filenames = f.read().splitlines()
        ind = 0
        for i in range(246):
            if 'region_'+str(i+1) in filenames[ind]:
                info = filenames[ind+1].split(' ')
                if info[0] != '0':
                    roi_size[i] = int(info[0])
            ind = ind+2

    return roi_size

def load_ADHD(Datadir, labelscsv, augmentation = 0):  ### labels: a cvs file
    with open(Datadir+'subjects.txt','r') as f:
        filenames = f.read().splitlines()


    fmri_signals = []

    for file in filenames:
        fmri_signal = np.load(Datadir + file + '.npy')
        # fmri_signal[fmri_signal<0]=0
        fmri_signals.append(fmri_signal)


    fmri_signals = np.stack(fmri_signals,axis=0)

    labels = []
    with open(labelscsv) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line = line + 1
                continue
            labels.append(row[1])


    return fmri_signals, labels, fmri_signals


def load_data(Datadir, labelscsv, augmentation = 0):  ### labels: a cvs file
    fmri_signals, DTI_net = data_reorder(Datadir)


    # graphs = np.zeros(DTI_net.shape)

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

    # train_fmri_signals, test_fmri_signals, train_labels, test_labels, train_graphs, test_graphs = train_test_split(fmri_signals,labels,graphs,test_size=0.2)
    # train_fmri_signals, val_fmri_signals, train_labels, val_labels, train_graphs, val_graphs = train_test_split(train_fmri_signals, train_labels, train_graphs, test_size=0.2)
    # # generate augmentation data
    #
    # train_fmri_net, train_labels , train_graph= augmentation_fmri_net(train_fmri_signals,train_labels,train_graphs,augmentation)
    # val_fmri_net, val_labels, val_graph = augmentation_fmri_net(val_fmri_signals, val_labels, val_graphs, 0)
    # test_fmri_net, test_labels , test_graph= augmentation_fmri_net(test_fmri_signals, test_labels, test_graphs, 0)
    #
    #
    #
    #
    # return train_fmri_net, train_graph, train_labels, val_fmri_net, val_graph, val_labels, test_fmri_net, test_graph, test_labels

    return fmri_signals, labels, DTI_net






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
    estimator = connectome.ConnectivityMeasure(kind='correlation')
    fmri_net = estimator.fit_transform([signal])[0]
    # fmri_net = np.abs(fmri_net)
    fmri_net = normalize_mat(fmri_net)
    # fmri_net[fmri_net<0]=0
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
    net = net - np.diag(np.diag(net))
    # flatten = np.matrix.flatten(abs(net))
    # sort = np.sort(flatten)
    # net = net/sort[-1]*1
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

def creat_csv_HCP(idtxt, allcsvlist,outcsv, keyword):
    with open(idtxt,'r') as f:
        filenames = f.read().splitlines()


    with open(outcsv,'w') as outf:
        spamwriter = csv.writer(outf,delimiter=',')
        spamwriter.writerow(['id','labels'])
        for file in filenames:
            for csv_s in allcsvlist:
                with open(csv_s, 'r') as f_csv:
                    csv_reader = csv.reader(f_csv,delimiter = '\t')
                    lind = 0
                    for line in csv_reader:
                        if lind ==0:
                            lind +=1
                            continue
                        row = line[0].split(',')
                        if file in row[0]:
                            if keyword == 'Gender':
                                if 'M' in row[3]: label = 1
                                else: label = 2
                            elif keyword == 'Age':
                                if '-' in row[4]:
                                    ages = row[4].split('-')
                                    label = (float(ages[0])+float(ages[1]))/2
                                elif '+' in row[4]:
                                    label = 36
                                else: label = float(row[4])
                            else:
                                try:
                                    label = float(row[127])
                                except:
                                    label = 0

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


def synthesis_data(Datadir, labelscsv, augmentation = 2000):  ### labels: a cvs file
    fmri_signals, DTI_net = data_reorder(Datadir)

    fmri_nets = np.zeros([fmri_signals.shape[0],fmri_signals.shape[2],fmri_signals.shape[2]])
    for ind in range(fmri_nets.shape[0]):
        fmri_nets[ind] = signals_to_net(fmri_signals[ind])

    mean_fmri_net = np.mean(fmri_nets[1:10],axis=0)
    # mean_fmri_net = fmri_nets[92]
    # # mean_fmri_net[mean_fmri_net > 0.35] = 1
    # mean_fmri_net[mean_fmri_net < 0.5] = 0



    mean_DTI_net = np.mean(DTI_net,axis=0)

    # mean_fmri_net = mean_DTI_net


    reg_id = [12, 215]
    reg_weight = [50, 100]


    pspnr = 16

    noise_level = np.max(mean_fmri_net)/pspnr




    syn_fmri_nets = []
    syn_DTI_nets = []
    labels = []
    for i in range(augmentation):
        noise = np.random.rand(mean_fmri_net.shape[0],mean_fmri_net.shape[1])
        noise = noise + noise.T
        noise = (noise-noise.min())/(noise.max()-noise.min())*noise_level

        injury_mat_A = np.ones(mean_fmri_net.shape)
        injury_A_score = (reg_weight[1]-reg_weight[0])*np.random.rand(1)+reg_weight[0]
        injury_vec_A = 1 + injury_A_score * np.random.uniform(0, 0.1, injury_mat_A.shape[0])
        injury_mat_A[reg_id[0], :] = injury_vec_A
        injury_mat_A[:, reg_id[0]] = injury_vec_A

        injury_mat_B = np.ones(mean_fmri_net.shape)
        injury_B_score = (reg_weight[1] - reg_weight[0]) * np.random.rand(1) + reg_weight[0]
        injury_vec_B = 1 + injury_B_score * np.random.uniform(0, 0.1, injury_mat_B.shape[0])
        injury_mat_B[reg_id[1], :] = injury_vec_B
        injury_mat_B[:, reg_id[1]] = injury_vec_B

        injury_mat_fmri = mean_fmri_net / (injury_mat_A * injury_mat_B)
        injury_mat_DTI = mean_DTI_net / (injury_mat_A * injury_mat_B)

        # np.fill_diagonal(injury_mat_fmri, 1)
        # np.fill_diagonal(injury_mat_DTI, 1)

        # id = random.getrandbits(1)
        # labels.append(id)
        id = 0
        if id==1:
            syn_fmri_nets.append(injury_mat_fmri+noise)
            syn_DTI_nets.append(injury_mat_DTI + noise)
            labels.append(injury_A_score)
        elif id==0:
            syn_fmri_nets.append(injury_mat_fmri + noise)
            syn_DTI_nets.append(injury_mat_DTI + noise)
            labels.append(injury_B_score)

    syn_fmri_nets = np.stack(syn_fmri_nets,axis=0)
    syn_DTI_nets = np.stack(syn_DTI_nets,axis=0)


    return syn_fmri_nets, syn_fmri_nets, labels

def probtrac2npy(Datadir, idtxt):
    with open(Datadir + '/' + idtxt,'r') as f:
        filenames = f.read().splitlines()

    for file in filenames:
        ROI_txt = Datadir+ '/' + file + '_ROI.txt'
        Data_connectivity = np.loadtxt(Datadir+'/'+file+'_DTI_net')
        if Data_connectivity.shape[0] != 246:

            with open(ROI_txt,'r') as f:
                # roi_size = np.ones([246],dtype=int)
                lines = f.read().splitlines()
                ind = 0
                # if Data_connectivity.shape[0]!=246:
                #     print(file + 'found!')
                if '133928' in file:
                    print ('pause')
                for i in range(246):
                    if 'region_'+str(i+1) in lines[ind]:
                        info = lines[ind+1].split(' ')
                        if info[0] == '0':
                            Data_connectivity = np.insert(Data_connectivity, i, 0, axis=0)
                            Data_connectivity = np.insert(Data_connectivity, i, 0, axis=1)
                        ind += 2
            if Data_connectivity.shape[0]!=246:
                print(file + ' failed')
                continue
        np.save(Datadir+'/'+file+'_DTInetworks.npy',Data_connectivity)


def read_ADHD(Datadir, Outdir):
    foldernames = ['KKI','NeuroIMAGE','NYU','OHSU','Peking_1','Peking_2','Peking_3','Pittsburgh','WashU']
    with open(Outdir + '/labels.csv', 'w') as outf:
        spamwriter = csv.writer(outf, delimiter=',')
        spamwriter.writerow(['id', 'labels'])

        for folder in foldernames:
            path = Datadir+'/'+folder
            csvf = Datadir+'/'+folder+'/'+folder+'_phenotypic.csv'
            with open(csvf) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                line = 0
                for row in csv_reader:
                    if line == 0:
                        line = line + 1
                        continue

                    if int(row[0])<100000: subjid='00'+row[0]
                    else: subjid = row[0]
                    if '0' != row[5]: label = 1
                    else: label = 0

                    subdata = []
                    find = False
                    for root, dirs, files in os.walk(path+'/'+subjid):
                        for file in files:
                            if 'sfnwmrda'+subjid in file and 'session_1_rest_1' in file:
                                ind = 0
                                with open(root+'/'+file,'r') as f:
                                    datafile = f.read().splitlines()
                                    for line in datafile:
                                        if ind ==0:
                                            ind+=1
                                            continue
                                        line = line.split('\t')
                                        subdata.append(np.asarray(line[2:92],dtype=float))
                                subdata=np.stack(subdata,axis=1)
                                netdata = signals_to_net(subdata.T)
                                np.save(Outdir+'/'+str(int(subjid))+'.npy',netdata)
                                find = True
                                spamwriter.writerow([subjid, label])
                                break
                    if not find: print(str(subjid))


def report_accuracy_metric(predic, labels):
    return precision_recall_fscore_support(predic,labels,average='binary')










if __name__ == '__main__':

    # ###-------------------------
    # dataset='HCP'
    # fmri_signals, labels, graphs = \
    #     load_data('Data_BNF/' + dataset + '/', 'Data_BNF/' + dataset + '/labels_gender.csv', 0)
    # import scipy.io as sio
    # data={}
    # data['fmri_net']=fmri_signals
    # data['dti_net']=graphs
    # data['label']=np.asarray(labels,np.int32)
    #
    # sio.savemat('array.mat', data)
    #
    # ###-------------------------

    import networkx as nx

    # dataset='HCP'
    # fmri_signals, labels, graphs = \
    #     load_data('Data_BNF/' + dataset + '/', 'Data_BNF/' + dataset + '/labels_gender.csv', 0)
    #
    # np.save('networks.npy',np.stack((fmri_signals,graphs),axis=0))

    Datadir='/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/'
    outdir='/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/node2vec-master/graph/HCP/'
    featdir='/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/gae/node2vec-master/emb/HCP/'
    net2edgelist(Datadir,outdir,featdir)




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

    # HCP
    # csvfiles = ['/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/HCP_behaviour.csv']
    # creat_csv_HCP('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/DTI_passed_subj.txt',
    #           csvfiles, '/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP/labels_PicVocab.csv','PicVocab_AgeAdj')

    # # ADHD
    # datadir = r'/home/local/ASUAD/wzhan139/Downloads/ADHD200_AAL_TCs_filtfix'
    # outdir = r'/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/ADHD'
    # read_ADHD(datadir,outdir)

    #--------------------------------------------------------------

    # train_fmri_net, train_adj, train_labels, val_fmri_net, val_adj, val_labels, test_fmri_net, test_adj, test_labels = \
    #     load_data('Data_BNF/Schiz/', 'Data_BNF/Schiz/labels.csv', 500)
    #
    # train_data = {}
    # train_data['fmri_net'] = train_fmri_net
    # train_data['graph'] = train_adj
    # train_data['label'] = train_labels
    #
    # val_data = {}
    # val_data['fmri_net'] = val_fmri_net
    # val_data['graph'] = val_adj
    # val_data['label'] = val_labels
    #
    # test_data = {}
    # test_data['fmri_net'] = test_fmri_net
    # test_data['graph'] = test_adj
    # test_data['label'] = test_labels
    #
    # data = {}
    # data['train'] = train_data
    # data['test'] = test_data
    # data['val'] =val_data
    #
    # pkl_file = open('Data_BNF/Schiz/sort_data.pkl', 'wb')
    # pickle.dump(data,pkl_file)
    # pkl_file.close()
    #--------------------------------------------------------------------------------------

    # ---------------------------------------------------------
    # probtrac2npy('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Data_BNF/HCP', 'DTI_passed_subj.txt')

    #----------------------------------------------------------
    # labels = np.load('Data_BNF/Schiz/result_cheby.npy')
    # precision, recall, f1_score, _ = precision_recall_fscore_support(labels[0], labels[1],average='weighted')
    # accuracy = accuracy_score(labels[0], labels[1])
    # print('accuracy is ', accuracy, '; precision is ', precision, '; recall is ', recall, '; f1-score is ', f1_score)

    #------

    # plt.figure()
    # fpr1, tpr1, thre1 = roc_curve(labels[1],labels[0])
    # auc1 = roc_auc_score(labels[0], labels[1])
    #
    # labels = np.load('Data_BNF/HCP/result_noweight.npy')
    # precision, recall, f1_score, _ = precision_recall_fscore_support(labels[0], labels[1], average='weighted')
    # accuracy = accuracy_score(labels[0], labels[1])
    # print('accuracy is ', accuracy, '; precision is ', precision, '; recall is ', recall, '; f1-score is ', f1_score)
    #
    # plt.figure()
    # fpr2, tpr2, thre2 = roc_curve(labels[1], labels[0])
    # auc2 = roc_auc_score(labels[0], labels[1])
    #
    #
    #
    # plt.plot(fpr1, tpr1, label='%s ROC (area = %0.2f)' % ('MDBN', auc1))
    # plt.plot(fpr2, tpr2, label='%s ROC (area = %0.2f)' % ('MDBN2', auc2))
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('1-Specificity(False Positive Rate)')
    # plt.ylabel('Sensitivity(True Positive Rate)')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()


    # #----------------------------------------------------------
    # dataset = 'Schiz'
    # fmri_signals, labels, graphs = \
    #     load_data('Data_BNF/' + dataset + '/', 'Data_BNF/' + dataset + '/labels.csv', 0)
    # male_mean_fmri = []
    # female_mean_fmri = []
    # male_mean_DTI = []
    # female_mean_DTI = []
    # for ind, x in enumerate(labels):
    #     if x=='1':
    #         male_mean_fmri.append(fmri_signals[ind])
    #         male_mean_DTI.append(graphs[ind])
    #     else:
    #         female_mean_fmri.append(fmri_signals[ind])
    #         female_mean_DTI.append(graphs[ind])
    #
    # male_mean_fmri=np.mean(male_mean_fmri,axis=0)
    # female_mean_fmri=np.mean(female_mean_fmri,axis=0)
    # male_mean_DTI = np.mean(male_mean_DTI, axis=0)
    # female_mean_DTI = np.mean(female_mean_DTI, axis=0)
    #
    # # labels = []
    # # net_female = np.zeros([48],dtype=np.float32)
    # # net_male = np.zeros([48],dtype=np.float32)
    #
    #
    # #-------- HCP
    # # labels = [36, 75, 93, 99, 108, 109, 114, 173, 197, 229]
    #
    # #-------- Schiz
    # labels = [45, 50, 109, 165, 168, 173, 187, 223, 224, 228]
    # male_mean_fmri=signals_to_net(male_mean_fmri)
    # female_mean_fmri=signals_to_net(female_mean_fmri)
    #
    #
    # # templatecsv='/mnt/easystore_8T/Wen_Data/Schizophrenia/Template/BNA_subregions.csv'
    # # with open(templatecsv) as csvfile:
    # #     csv_reader = csv.reader(csvfile, delimiter=',')
    # #     line = 0
    # #     abbr=''
    # #     for row in csv_reader:
    # #         if line == 0:
    # #             line = line + 1
    # #             continue
    # #         if not row[1]=='':
    # #             abbr=row[1].split(',')[0]
    # #             labels.append(abbr+'_L')
    # #             labels.append(abbr+'_R')
    # #             line+=2
    # #         net_female[line-3]+=female_mean_fmri[int(row[3])-1,114]
    # #         net_female[line-2]+=female_mean_fmri[int(row[4])-1,114]
    # #         net_male[line - 3] += male_mean_fmri[int(row[3]) - 1, 114]
    # #         net_male[line-2] += male_mean_fmri[int(row[4]) - 1, 114]
    # # net_female=net_female/np.min(abs(net_female))*10
    # # net_male=net_male/np.min(abs(net_male))*10
    #
    #
    # female_net=np.zeros([10,10])
    # male_net = np.zeros([10, 10])
    # for i in range(10):
    #     for j in range(i+1,10):
    #         female_net[i,j]=female_mean_fmri[labels[i],labels[j]]
    #         female_net[j,i]=female_net[i,j]
    #         male_net[i, j] = male_mean_fmri[labels[i], labels[j]]
    #         male_net[j, i] = male_net[i, j]
    #
    # female_net*=200
    # male_net*=200
    #
    # female_DTI_net = np.zeros([10, 10])
    # male_DTI_net = np.zeros([10, 10])
    # for i in range(10):
    #     for j in range(i + 1, 10):
    #         female_DTI_net[i, j] = female_mean_DTI[labels[i], labels[j]]
    #         female_DTI_net[j, i] = female_DTI_net[i, j]
    #         male_DTI_net[i, j] = male_mean_DTI[labels[i], labels[j]]
    #         male_DTI_net[j, i] = female_DTI_net[i, j]
    #
    # female_DTI_net *= 200
    # male_DTI_net *= 200
    #
    # # female_DTI_net[female_DTI_net<1]=0
    # # male_DTI_net[male_DTI_net > 100] = 0
    #
    #
    # with open('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GAT/Result/female_fmri_net_schiz.txt','w') as  f:
    #     f.write('labels')
    #     for i in range(10):
    #         f.write(' '+str(labels[i]))
    #     f.write('\n')
    #
    #     for i in range(10):
    #         f.write(str(labels[i]))
    #         for j in range(10):
    #             f.write(' '+str(female_net[i,j]))
    #         f.write('\n')
    #
    #
    #
    #
    #
    #
    # # plt.matshow(male_mean_fmri)
    # # plt.matshow(female_mean_fmri)
    # # plt.show()


    #---------------------------------------------------------- original vs reconstruction stat plot
    # data = np.stack([wz.flatten(), wz2.flatten()], axis=1)
    # df = pd.DataFrame(data, columns=["True Functional Connectivity", "Reconstructed Functional Connectivity"])
    # sns.jointplot(data=df, x='True Functional Connectivity', y='Reconstructed Functional Connectivity', kind='reg',
    #               joint_kws={'color': 'green', 'scatter_kws': {'s': 1}, 'line_kws': {'color': '#ff459f'}})