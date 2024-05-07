import os
os.system("pip install -r "+os.path.dirname(os.path.abspath(__file__))+"\\requirements.txt")
print("-------\tSelect Dataset-------")
print("1. Cora")
print("2. Citeseer")
print("3. Pubmed")
print("4. CS")
print("5. Physics")
print("6. DBLP")
temp_selection = int(input("Selection Number (Type 1 for Cora): "))
if temp_selection == 1:
    dataset_global = "Cora"
elif temp_selection == 2:
    dataset_global = "Citeseer"
elif temp_selection == 3:
    dataset_global = "Pubmed"
elif temp_selection == 4:
    dataset_global = "CS"
elif temp_selection == 5:
    dataset_global = "Physics"
elif temp_selection == 6:
    dataset_global = "DBLP"
else:
    print("Error in option selection. Going ahead with default Cora")
    dataset_global = "Cora"
print("-------\tSelect Coarsening Ratio-------")
if dataset_global in ["Cora","Citeseer"]:
    print("The following Corseing ratios are available. ")
    print("0.3\n0.1\n0.05")
    r_global = float(input("Enter the coarsening ratio (Type 0.3 for coarsening ratio of 0.3): "))
    if r_global not in [0.3,0.1,0.05]:
        print("Error in selection going ahead with default 0.05")
        r_global = 0.1
elif dataset_global in ["Pubmed","Physics","CS","DBLP"]:
    print("The following Corseing ratios are available. ")
    print("0.03\n0.01\n0.05")
    r_global = float(input("Enter the coarsening ratio (Type 0.3 for coarsening ratio of 0.3): "))
    if r_global not in [0.03,0.01,0.05]:
        print("Error in selection, going ahead with default 0.05")
        r_global = 0.05
print("-------\tSelect Algorithm-------")
print("1. LAGC")
print("2. FGC")
algo_global = input("Selection Number (Enter 1 for LAGC): ")
if algo_global == "1":
    algo_global = "LAGC"
elif algo_global == "2":
    algo_global = "FGC"
else:
    print("Error in selection. Going ahead with default LAGC")
    algo_global = "LAGC"


k_global = 0
exp_iter = 10


from networkx.algorithms import community
from networkx.generators.community import random_partition_graph
from networkx.generators.community import stochastic_block_model
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.random_graphs import watts_strogatz_graph
from random import sample
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy.sparse import random
from scipy.sparse import random
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import norm
from scipy.stats import rv_continuous
from sklearn.decomposition import FactorAnalysis
from torch import Tensor
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import DBLP
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import WebKB
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse,homophily
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from tqdm import tqdm
import collections
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
# from pyswarm import pso
import scipy.sparse as sp
import seaborn as sns
import torch.nn.functional as F
import torch_geometric
import torch
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("-------\tLoading Dataset "+ dataset_global +"-------")
if dataset_global in ['CS','Physics']:
    dataset = Coauthor(root=os.path.dirname(os.path.abspath(__file__))+'/data/Coauthor',name=dataset_global)
elif dataset_global in ['Cora','Citeseer','Pubmed']:
    dataset = Planetoid(root=os.path.dirname(os.path.abspath(__file__))+'/data/Planetoid',name=dataset_global)
elif dataset_global in ['DBLP']:
    dataset = CitationFull(root=os.path.dirname(os.path.abspath(__file__))+'/data/DBLP',name='DBLP')
else:
    print("No Dataset Provided. Going with DBLP")
    dataset = CitationFull(root=os.path.dirname(os.path.abspath(__file__))+'/data/DBLP',name='DBLP')
print("-------\tSetting up output files-------")
try:
    os.makedirs(os.path.dirname(os.path.abspath(__file__))+"/"+algo_global +"/"+dataset_global+"/"+str(r_global))
    output_path = os.path.dirname(os.path.abspath(__file__))+"/"+algo_global +"/"+dataset_global+"/"+str(r_global)
except:
    print("Output directory already exists.")

print("-------\tLoading Parameters-------")
df_parameters = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/Parameters_Fine_Tuned.csv')
df_parameters = df_parameters[df_parameters['Dataset']==dataset_global]
df_parameters = df_parameters[df_parameters['Ratio']==r_global]
df_parameters = df_parameters[df_parameters['Experiment']==algo_global]
print(df_parameters)
alpha_param,beta_param,gamma_param,lambda_param,delta_param = df_parameters[['Alpha','Beta','Gamma','Lambda','Delta']].values[0]









edge_list = dataset[0].edge_index
NO_OF_EDGES = edge_list.shape[1]
labels = dataset[0].y

# print("Homophilic ratio : " + str(homophily(edge_list,labels,method='edge')))


adj = to_dense_adj(dataset[0].edge_index)
adj = adj[0]

labels = labels.numpy()

X = dataset[0].x
X = X.to_dense()
N = X.shape[0]
NO_OF_CLASSES = len(set(labels))

sparsity_original = 2*NO_OF_EDGES/(N*(N-1))
# print("Sparsity of original graph : " + str(sparsity_original))

nn = int(1*N)
X = X[:nn,:]
adj = adj[:nn,:nn]
labels = labels[:nn]
# print(X.shape,adj.shape)



def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

theta = get_laplacian(adj)
# print(theta.shape)


features = X.numpy()
NO_OF_NODES = X.shape[0]

total_indices1 = torch.arange(NO_OF_NODES)
shuffled_indices1 = torch.randperm(total_indices1.numel())
train_mask = int(0.1 * total_indices1.numel())
#print("Train Mask", train_mask)


def convertScipyToTensor(coo):
    try:
        coo = coo.tocoo()
    except:
        coo = coo
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


from scipy.sparse import csr_matrix
from scipy.sparse import random
from scipy.sparse.linalg import norm
from scipy.stats import rv_continuous

p = X.shape[0]
k = int(p*r_global)
k_global = int(p*r_global)
n = X.shape[1]
lambda_param = 100
beta_param = 50
alpha_param = 100
gamma_param = 100
lr = 1e-5
thresh = 1e-10


class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()  # get a frozen version of the distribution
X_tilde = random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
C = random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)

test_mask = total_indices1.numel()
total_indices1.numel()


def experiment(alpha_param,beta_param,gamma_param,lambda_param,delta_param,C,X_tilde,theta,X):
    p = X.shape[0]
    k = int(p*r_global)
    n = X.shape[1]
    ones = csr_matrix(np.ones((k,k)))
    ones = convertScipyToTensor(ones)
    ones = ones.to_dense()
    J = np.outer(np.ones(k), np.ones(k))/k
    J = csr_matrix(J)
    J = convertScipyToTensor(J)
    J = J.to_dense()
    zeros = csr_matrix(np.zeros((p,k)))
    zeros = convertScipyToTensor(zeros)
    zeros = zeros.to_dense()
    X_tilde = convertScipyToTensor(X_tilde)
    X_tilde = X_tilde.to_dense()
    C = convertScipyToTensor(C)
    C = C.to_dense()
    eye = torch.eye(k)
    try:
        theta = convertScipyToTensor(theta)
    except:
        theta = theta
    try:
        X = convertScipyToTensor(X)
        X = X.to_dense()
    except:
        X = X

    def one_hot(x, class_count):
        return torch.eye(class_count)[x, :]

    P = labels
    P = one_hot(P,NO_OF_CLASSES)
    P[train_mask,:]=0
    if(torch.cuda.is_available()):
        # print("yes")
        X_tilde = X_tilde.cuda()
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        J = J.cuda()
        P = P.cuda()
        zeros = zeros.cuda()
        ones = ones.cuda()
        eye = eye.cuda()
    def update(X_tilde,C,i):
        global L
        thetaC = theta@C
        CT = torch.transpose(C,0,1)
        X_tildeT = torch.transpose(X_tilde,0,1)
        CX_tilde = C@X_tilde
        t1 = CT@thetaC + J
        term_bracket = torch.linalg.pinv(t1)
        thetacX_tilde = thetaC@(X_tilde)

        L = 1/k

        t1 = -2*gamma_param*(thetaC@term_bracket)
        t2 = alpha_param*(CX_tilde-X)@(X_tildeT)
        t3 = 2*thetacX_tilde@(X_tildeT)
        t4 = lambda_param*(C@ones)
        t5 = 2*beta_param*(thetaC@CT@thetaC)
        t6 = delta_param*P@torch.transpose((CT@P),0,1)
        T2 = (t1+t2+t3+t4+t5+t6)/L
        Cnew = (C-T2).maximum(zeros)
        t1 = CT@thetaC*(2/alpha_param)
        t2 = CT@C
        t1 = torch.linalg.pinv(t1+t2)
        t1 = t1@CT
        t1 = t1@X
        X_tilde_new = t1
        Cnew[Cnew<thresh] = thresh
        for i in range(len(Cnew)):
            Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i],1)
        for i in range(len(X_tilde_new)):
            X_tilde_new[i] = X_tilde_new[i]/torch.linalg.norm(X_tilde_new[i],1)
        return X_tilde_new,Cnew


    for i in tqdm(range(20)):
        X_tilde,C = update(X_tilde,C,i)

    return X_tilde,C

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 64)
        self.conv2 = GCNConv(64, NO_OF_CLASSES)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        #print("Checking 1: x", x.shape, "Edge index:", edge_index.shape)
        x = self.conv1(x, edge_index)
        #print("Checking 2: convolution done, new x:", x.shape)
        x = F.relu(x)
        #print("Checking 3: x", x.shape, "training:", self.training)
        x = F.dropout(x, training=self.training)
        #print("Checking 4: dropout done new x", x.shape, "Edge index:", edge_index.shape)
        x = self.conv2(x, edge_index)
        #print("Checking 5: x", x.shape)

        return F.log_softmax(x, dim=1)
    
from random import sample
from torch_geometric.utils import dense_to_sparse,homophily



def get_accuracy(C_0,L,X_t_0):
    gc.collect()
    global labels, NO_OF_CLASSES,k
    all_acc = []
    for i in [1,2]:
        C_0_new=np.zeros(C_0.shape)
        for i in range(C_0.shape[0]):
            C_0_new[i][np.argmax(C_0[i])]=1
        # print("C_0_new:",C_0_new)
        # C_0_new=C_0
        from scipy import sparse

        Lc=C_0_new.T@L@C_0_new

        Wc=(-1*Lc)*(1-np.eye(Lc.shape[0]))
        Wc[Wc<0.1]=0
        Wc=sparse.csr_matrix(Wc)
        Wc = Wc.tocoo()
        row = torch.from_numpy(Wc.row).to(torch.long)
        col = torch.from_numpy(Wc.col).to(torch.long)
        edge_index_coarsen2 = torch.stack([row, col], dim=0)


        def one_hot(x, class_count):
            return torch.eye(class_count)[x, :]

        device = torch.device('cpu')

        Y = labels

        Y = one_hot(Y,NO_OF_CLASSES)
        Y[train_mask, :] = 0
        # making Training dataset
        # Y[dataset[0].train_mask, :] = 0



        P=np.linalg.pinv(C_0_new)
        labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double() , Y.double()).double() , 1)

        Wc=Wc.toarray()
        adjtemp = torch.tensor(Wc)
        # edge_list_temp = dense_to_sparse(adjtemp)[0]

        # print("Homophilic ratio : " + str(homophily(edge_list_temp,labels_coarse,method='edge')))
        # number_of_edges = edge_list_temp.shape[1]
        # n = labels_coarse.shape[0]
        # sparsity = 2*number_of_edges/(n*(n-1))
        # print("Sparsity : " + str(sparsity))

  #
        # C2=np.linalg.pinv(C_0_new)
        model=Net().to(device)
        device = torch.device('cpu')
        lr=0.01
        decay=0.0001
        try:
            X=np.array(features.todense())
        except:
            X = np.array(features)
        #print("X:",X.shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        # criterion=torch.nn.CrossEntropyLoss()
        x=sample(range(0, int(k)), k)

        from datetime import datetime
        Xt=P@X
        # Xt=X_t_0
        #########################################################  adding masks
        total_indices = torch.arange(k_global)
        shuffled_indices = torch.randperm(total_indices.numel())
        train_size = int(0.8 * total_indices.numel())
        train_indices = shuffled_indices[:train_size]
        val_size = int(0.2 * total_indices.numel())
        val_indices = shuffled_indices[train_size:train_size + val_size]

        train_indices = shuffled_indices[:]

        data = dataset[0].to(device)

        coarsen_features = torch.Tensor(Xt).to(device)
        coarsen_train_labels = labels_coarse.to(device)
        coarsen_train_mask = train_indices.to(device)
        coarsen_val_labels = labels_coarse.to(device)
        coarsen_val_mask = val_indices.to(device)
        coarsen_edge = edge_index_coarsen2.to(device)

        # print("Coarsen Data _trainpy", data)
        # print("Coarsen feature _trainpy", coarsen_features.shape)
        # print("Coarsen train_labels _trainpy", coarsen_train_labels.shape)
        # print("Coarsen train_mask _trainpy", coarsen_train_mask)
        # print("Coarsen train_mask _trainpy", torch.sum(coarsen_train_mask))
        # print("Coarsen val_mask _trainpy", torch.sum(coarsen_val_mask))


        # if args.normalize_features:
        #     coarsen_features = F.normalize(coarsen_features, p=1)
        #     data.x = F.normalize(data.x, p=1)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        pathm = os.path.dirname(os.path.abspath(__file__))+"/"+algo_global +"/"+dataset_global+"/"+str(r_global)+"/"
        best_val_loss = float('inf')
        no_of_epochs = 500
        no_of_early_stopping = 10
        val_loss_history = []
        for epoch in range(no_of_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(coarsen_features, coarsen_edge)
            loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            pred = model(coarsen_features, coarsen_edge)
            val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

            if val_loss < best_val_loss and epoch > no_of_epochs // 2:
                best_val_loss = val_loss
                torch.save(model.state_dict(), pathm + 'checkpoint-best-acc.pkl')

            val_loss_history.append(val_loss)
            if no_of_early_stopping  > 0 and epoch > no_of_epochs // 2:
                tmp = torch.Tensor(val_loss_history[-(no_of_early_stopping  + 1):-1])
                if val_loss > tmp.mean().item():
                    break


        model.load_state_dict(torch.load(pathm + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred = model(data.x, data.edge_index).max(1)[1]
        #test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
        test_acc = int(pred.eq(data.y).sum().item()) / int(test_mask)
        print("Test_acc",f"{test_acc:.4f}")
        all_acc.append(test_acc)

    print('ave_acc: {:.8f}'.format(np.mean(all_acc)), '+/- {:.8f}'.format(np.std(all_acc)))
    gc.collect()
    return np.mean(all_acc)


def getSparsityAndHomophily(C,theta):
    theta = C.T@theta@C
    adjtemp = -theta
    for i in range(adjtemp.shape[0]):
        adjtemp[i,i]=0
    adjtemp[adjtemp<0.01]=0
    temp = dense_to_sparse(adjtemp)
    edge_list_temp = temp[0]
    # ytemp = temp[1]
    # P = torch.linalg.pinv(C)
    # labels =
    # # print(edge_list)
    number_of_edges = edge_list_temp.shape[1]
    # n = adjtemp.shape[0]

    # print("Homophilic ratio : " + str(homophily(edge_list_temp,ytemp,method='node')))
    # sparsity = 2*number_of_edges/(n*(n-1))
    # print("Sparsity : " + str(sparsity))


def fitness_function(alpha_param,beta_param,gamma_param,lambda_param,delta_param):
    print(alpha_param,beta_param,gamma_param,lambda_param,delta_param)
    #alpha_param,beta_param,gamma_param,lambda_param,delta_param = temp_param
    X_tilde = random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs)
    C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
    X_t_0,C_0 = experiment(alpha_param,beta_param,gamma_param,lambda_param,delta_param,C,X_tilde,theta,X)
    L = theta
    #getSparsityAndHomophily(C_0,theta)

    C_0 = C_0.cpu().detach().numpy()
    X_t_0 = X_t_0.cpu().detach().numpy()
    try:
        L = L.cpu().detach().numpy()
    except:
        L = L
    acc = get_accuracy(C_0,L,X_t_0)
    #readings_path 
    return acc





try:
    alpha_param,beta_param,gamma_param,lambda_param,delta_param = df_parameters[['Alpha','Beta','Gamma','Lambda','Delta']].values[0]
    acc = fitness_function(alpha_param,beta_param,gamma_param,lambda_param,delta_param)
    gc.collect()
except Exception as e:
    print(e)
