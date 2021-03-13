import os

import sklearn.model_selection
import sslbookdata
from sslbookdata.sslbookdata import load_secstr

import numpy as np
import os.path as osp
import pandas as pd
from tf_labelprop.gssl.graph.gssl_utils import init_matrix
import tf_labelprop.gssl.graph.gssl_utils as gutils
from tf_labelprop.input.dataset import GSSLDataset, download_url, extract_zip
from tf_labelprop.input.dataset._mnist import INPUT_FOLDER


def getDataframe(folder,ds_name):
    path_X = osp.join(folder,ds_name + "_X.csv")
    path_Y = osp.join(folder,ds_name + "_Y.csv")
    
    X = pd.read_csv(path_X,sep=",",index_col=0,header=0)
    Y = pd.read_csv(path_Y,sep=",",index_col=0,header=0)
    return {"X":X.values,"Y":np.reshape(Y.values,(-1)) - 1}


class ISOLET(GSSLDataset):
    
    @staticmethod
    def get_name(self):
        return "ISOLET"
    
    
    @property
    def processed_file_names(self):
        return [""]
        
    @property
    def raw_file_names(self):
        return  ["phpB0xrNj"]
    
    def download(self):
            download_url("https://www.openml.org/data/get_csv/52405/phpB0xrNj",
                          self.raw_dir)
            
            
    def process(self):
        pass

        
        
    def load(self):
        GSSLDataset.load(self)
        
        X = pd.read_csv(osp.join(self.raw_dir,self.raw_file_names[0]))
        Y = np.reshape(X['class'].astype(np.str),(-1,)).values
        Y =  np.array([int(y.strip("'")) - 1 for y in Y])
        X = X.drop(['class'],axis=1).values
        
        assert X.shape == (7797,617)
        W = None
        return X,W,Y
                                   
        
    
    def __init__(self,root=INPUT_FOLDER):
        root = os.path.join(root,"isolet")
        super(ISOLET,self).__init__(root)
        

class ChapelleDataset(GSSLDataset):
    @staticmethod
    def get_name(self):
        return "Chapelle"
    @property
    def name(self):
        return self.dataset
    
    
    @property
    def processed_file_names(self):
        return [""]
        
    @property
    def raw_file_names(self):
        return  [""]
    
    def download(self):
        pass
            
            
    def process(self):
        pass

        
    labeledIndexes = None
    
    def load(self):
        GSSLDataset.load(self)
        
        def load_secstr(split,labels,extra_unlabeled=False):
            def explode(Xin):
                m, d0 = Xin.shape
                ks = np.unique(Xin)
                k = len(ks)
                d1 = k * d0
                X = np.zeros((m,d1), dtype='u1')
                l = 0
                for i in range(k):
                    X[:, l:l+d0] = Xin == ks[i]
                    l = l + d0
                return X
            
            mat = sslbookdata.sslbookdata.pkg_loadmat('data/data8.mat')

            if np.min(mat['y']) == -1:
                mat['y'][mat['y'] == -1] = 0
            
            mat['y'] = init_matrix(Y=np.reshape(mat['y'],(-1,)) ,labeledIndexes=[True]*mat['y'].shape[0])
            
            
            mat['X'] = explode(mat['T'])
            
            if extra_unlabeled:
                mat2 = sslbookdata.sslbookdata.pkg_loadmat('data/data8extra.mat')
                Xextra = explode(mat2['T'])
                yextra_shape = (Xextra.shape[0],mat['y'].shape[1])
                mat['y'] = np.concatenate([mat['y'],np.zeros(yextra_shape) ],axis=0)
                mat['X'] = np.concatenate([mat['X'],Xextra],axis=0)
            
            allsplits = sslbookdata.sslbookdata.pkg_loadmat(f'data/splits8-labeled{labels}.mat')
            labeledIndexes = allsplits['idxLabs'][split, :] - 1
            
            return mat, labeledIndexes
            
        def loadmat(ds_id,split,labels):
            mat = sslbookdata.sslbookdata.pkg_loadmat(f'data/data{ds_id}.mat')
            
            if np.min(mat['y']) == -1:
                    mat['y'][mat['y'] == -1] = 0
                
            mat['y'] = init_matrix(Y=np.reshape(mat['y'],(-1,)) ,labeledIndexes=[True]*mat['y'].shape[0])
                
            
            if self.use_splits:
                try:
                    allsplits = sslbookdata.sslbookdata.pkg_loadmat(f'data/splits{ds_id}-labeled{labels}.mat')
                    labeledIndexes = allsplits['idxLabs'][split, :] - 1
                except:
                    raise ValueError("Could not load splits")
                
                
                return mat, labeledIndexes
            else:
                return mat, None
            
        
        dct = { "digit1": lambda split,labels: loadmat(1,split,labels),
                "USPS": lambda split,labels: loadmat(2,split,labels),
                "COIL2": lambda split,labels: loadmat(3,split,labels),
                "BCI": lambda split,labels: loadmat(4,split,labels),
                "g241c": lambda split,labels: loadmat(5,split,labels),
                "COIL": lambda split,labels: loadmat(6,split,labels),
                "g241n": lambda split,labels: loadmat(7,split,labels),
                "SecStrExtra": lambda split,labels: load_secstr(split,labels,extra_unlabeled=True),
                "SecStr": lambda split,labels: load_secstr(split,labels,extra_unlabeled=False),
                "Text": lambda split,labels: loadmat(1,split,labels),
            }
        res, labeledIndexes = dct[self.dataset](self.split if self.use_splits else 0,self.labels)
        X, Y = res['X'], res['y']
        if not labeledIndexes is None:
            self.labeledIndexes = np.array([False]*X.shape[0])
            self.labeledIndexes[labeledIndexes] = True
        W = None
        


        return np.array(X),W,np.array(Y)
                                   
        
    
    def __init__(self,benchmark,labels=100,split=0,use_splits=False):
        super(ChapelleDataset,self).__init__()
        self.dataset = benchmark
        self.labels = labels
        self.split = split
        self.use_splits = use_splits
        
        


if __name__ == "__main__":
    
    for x in ["SecStrExtra"]:
        for i in range(10):
            ds = ChapelleDataset(x,split=i)
            X,W,Y = ds.load()
    #print(X.shape)
    #print(Y.shape)
