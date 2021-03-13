'''
Created on 22 de Nov de 2019

@author: klaus
'''
import os
import pickle

import numpy as np
import os.path as osp
import pandas as pd
from tf_labelprop.input.dataset import download_url, extract_gz, GSSLDataset
from tf_labelprop.logging  import logger as LOG
from tf_labelprop.settings import INPUT_FOLDER


def convert(imgf, labelf, outf, n):
    import gzip
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()



def get_mnist(MNIST_FOLDER):
    if not osp.isfile(osp.join(MNIST_FOLDER,'mnist_train.csv')):
        """
        zip_path = osp.join(MNIST_FOLDER,'mnist_data.zip')
        if not osp.isfile(zip_path):
            raise ValueError("Could not find MNIST data at {}".format(zip_path))
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            LOG.info("Extracting MNIST...")
            zip_ref.extractall(MNIST_FOLDER)
        """
        def get_path(fname):
            return osp.join(MNIST_FOLDER,fname)
        
        for f in ["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]:
            
            download_url("http://yann.lecun.com/exdb/mnist/{}".format(f),
                          MNIST_FOLDER)
            extract_gz(osp.join(MNIST_FOLDER,f))
        convert(osp.join(MNIST_FOLDER,"train-images-idx3-ubyte"),
                osp.join(MNIST_FOLDER,"train-images-idx3-ubyte"),
                osp.join(MNIST_FOLDER,"mnist_train.csv"),
                60000)
        convert(osp.join(MNIST_FOLDER,"t10k-images-idx3-ubyte"),
                osp.join(MNIST_FOLDER,"t10k-labels-idx1-ubyte"),
                osp.join(MNIST_FOLDER,"mnist_test.csv"), 10000)
    
    
    X = pd.concat([pd.read_csv(os.path.join(MNIST_FOLDER,'mnist_train.csv'),header=None),
                   pd.read_csv(os.path.join(MNIST_FOLDER,'mnist_test.csv'),header=None)],axis=0).values
    Y = np.reshape(X[:,0],(-1,))
    X = X[:,1:]
                                         
    
    return {"X":X,"Y": Y}





class MNIST(GSSLDataset):
    
    name = "MNIST"
    
    @property
    def processed_file_names(self):
        return ["mnist_train.csv","mnist_test.csv"]
        
    @property
    def raw_file_names(self):
        return  ["train-images-idx3-ubyte","train-labels-idx1-ubyte",
                  "t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte"]
    
    def download(self):
        for f in ["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]:
                
            download_url("http://yann.lecun.com/exdb/mnist/{}".format(f),
                          self.raw_dir)
            fpath = osp.join(self.raw_dir,f)
            fpath_extracted = fpath[:-3]
            extract_gz(fpath,fpath_extracted)
            
            
    def process(self):
        convert(osp.join(self.raw_dir,"train-images-idx3-ubyte"),
                osp.join(self.raw_dir,"train-labels-idx1-ubyte"),
                osp.join(self.processed_dir,"mnist_train.csv"),
                60000)
        convert(osp.join(self.raw_dir,"t10k-images-idx3-ubyte"),
                osp.join(self.raw_dir,"t10k-labels-idx1-ubyte"),
                osp.join(self.processed_dir,"mnist_test.csv"), 10000)
        
        
    def load(self):
        GSSLDataset.load(self)
        X = pd.concat([pd.read_csv(os.path.join(self.processed_dir,'mnist_train.csv'),header=None),
                   pd.read_csv(os.path.join(self.processed_dir,'mnist_test.csv'),header=None)],axis=0).values
        Y = np.reshape(X[:,0],(-1,))
        
        X = X[:,1:]
        W = None
        return X,W,Y
                                   
        
    
    def __init__(self,root=INPUT_FOLDER):
        root = os.path.join(root,"mnist_data")
        super(MNIST,self).__init__(root)
        
        
        
if __name__ == "__main__":
    X, W, Y= MNIST().load()
    print(X.shape)



