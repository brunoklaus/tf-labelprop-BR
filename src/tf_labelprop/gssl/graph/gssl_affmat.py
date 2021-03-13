"""
gssl_affmat.py
====================================
Module that handles the construction of affinity matrices.
"""

from functools import partial
import os
import time

import faiss
import scipy.sparse
from sklearn.neighbors import NearestNeighbors

import numpy as np
import os.path as osp
import scipy.spatial.distance as scipydist
from tf_labelprop.gssl.graph.gssl_utils import lap_matrix, extract_lap_eigvec
import tf_labelprop.gssl.graph.gssl_utils as  gutils
import tf_labelprop.logging.logger as LOG
from tf_labelprop.settings import load_sparse_csr


#import sys
#import progressbar
def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    s_tuple = sorted(tuples, key=lambda x: (x[2], x[1]))
    
    row,col,data  = zip(*s_tuple)
    return scipy.sparse.coo_matrix((data, (row,col)), shape=m.shape)
class AffMat(scipy.sparse.csr_matrix):
    
    from tf_labelprop.gssl.graph.gssl_utils import extract_lap_eigvec,lap_matrix
    @staticmethod
    def cache_mat_exists(cache_dir):
        if cache_dir is None:
            return False
        return osp.isfile(osp.join(cache_dir,f'affmat.npz'))
    
    def load_eigenfunctions(self,m,which_lap='sym',D=None,remove_first_eig=False):
        """ Extract  ``m`` eigenvectors and eigenvalues of the laplacian, in non-decreasing order. 
        
            Args:
                which_lap (str) : Chooses the type of laplacian. One of `sym`,`comb` or `rw`.           
                m (int) : number of eigenvectors to extract
                D (`[NDArray[float].shape[N,N]`) : extra matrix for generalized eigenvalue problem
    
            
            Returns:
                Pair[NDArray[float].shape[M,N],NDArray[float].shape[M]] : matrix of eigenvectors, and vector of eigenvalues
                
        """
        if not self.cache_dir is None:
            eigvec_path = osp.join(self.cache_dir,f'eigvec_{which_lap}.npy')
            eigval_path = osp.join(self.cache_dir,f'eigval_{which_lap}.npy')
            files_exist = (osp.isfile(eigvec_path)) and (osp.isfile(eigval_path))
            if files_exist:
                LOG.info(f"Loading eigenfunctions in {eigvec_path} ...")
                EIGVAL = np.load(eigval_path)
                EigVec = np.load(eigvec_path)
                if EIGVAL.shape[0] >= m:
                    return EigVec[:,:m], EIGVAL[:m]

        
        L = lap_matrix(self,which_lap)
        print(f"Extracting {m} eigenvectors for matrix L (shape: {L.shape}, #edges= {L.data.shape}")
        m = min(m,L.shape[0]-1)
        eigVec, eigVal = extract_lap_eigvec(L,m,D,remove_first_eig)
        if (not self.cache_dir is None):
            LOG.info(f"Saving  eigenfunctions to {eigvec_path} ...")
            np.save(eigvec_path,eigVec)
            np.save(eigval_path,eigVal)
        return eigVec, eigVal
           
            
    
    def __init__(self,W=None,cache_dir=None):
        self.cache_dir = cache_dir
        if (not self.cache_dir is None):
            if not osp.isdir(self.cache_dir):
                os.makedirs(self.cache_dir)
    
            aff_path = osp.join(self.cache_dir,'affmat.npz')
            if AffMat.cache_mat_exists(cache_dir):            
                if osp.isfile(aff_path):
                    W = scipy.sparse.load_npz(aff_path)
                    
                    
            
        if W is None:
            raise ValueError(f"Error: provided no affinity matrix and/or could not load from cache = {self.cache_dir}")
        if not self.cache_dir is None:
            scipy.sparse.save_npz(aff_path,W)
            
        super(scipy.sparse.csr_matrix, self).__init__(W)


class AffMatGenerator(object):
    """Constructs an affinity matrix from some specification.
    """
    
    
    def get_or_calc_Mask(self,X):
        """ Gets the previously computed mask for affinity matrix, or computes it."""
        if self.K is None:
            self.K = self.mask_func(X)
        return(self.K.astype(np.double))
    
    
    def handle_adaptive_sigma(self,K):
        if not scipy.sparse.issparse(K):
            M = K
            M[M==0] = np.infty 
            M = np.sort(M, axis=1)
            self.sigma = np.mean(M[:,9])/3
            LOG.info("Adaptive sigma is {}".format(self.sigma),LOG.ll.MATRIX)
        else:
            self.sigma = np.mean([np.sort(K.getrow(i).data)[9]/3 for i in range(K.shape[0])])
        return partial(lambda d: np.exp(-(d*d)/(2*self.sigma*self.sigma)))
    
    def W_from_K(self,X,K):
        if not scipy.sparse.issparse(K):
            if self.dist_func_str == "LNP" or self.dist_func_str == "NLNP":
                W = self.dist_func(X,K)
            else:
                W =  np.reshape([0 if x == 0 else self.dist_func(x) for x in  np.reshape(K,(-1))],K.shape)
        else:
            if self.dist_func_str == "LNP" or self.dist_func_str == "NLNP":
                raise NotImplementedError("Did not implement LNP on sparse matrix yet")
            else:
                W = scipy.sparse.csr_matrix(K)
                W.data = np.asarray([self.dist_func(x) for x in W.data])
        return W  
    
    
    def generateAffMat(self,X,Y=None,labeledIndexes=None,hook=None):
        """ Generates the Affinity Matrix.
        
            Returns:
                `tflabelprop.gssl.graph.gssl_affmat.AffMat`: An affinity matrix
         """
         
        """
             Return Cached matrix, if cache directory exists
        """
        X = X.astype(np.float32)
        
        if AffMat.cache_mat_exists(self.cache_dir):
            LOG.info(f"Loading Affinity Matrix from {self.cache_dir}...",LOG.ll.MATRIX)
            return AffMat(W=None,cache_dir=self.cache_dir)
         
        LOG.info("Creating Affinity Matrix...",LOG.ll.MATRIX)
        
        if not hook is None:
            hook._begin(X=X,Y=Y,labeledIndexes=labeledIndexes,W=None)
        
        K = self.get_or_calc_Mask(X)
        
        if self.sigma == "mean":
            self.dist_func = self.handle_adaptive_sigma(K)
        

        if not K.shape[0] == X.shape[0]:
            raise ValueError("Shapes do not match for X,K")
            
        
        W = self.W_from_K(X,K)
        
        if self.row_normalize == True:
            W = gutils.deg_matrix(W, pwr=-1.0, NA_replace_val=1.0) @ W 
        
        del K
        LOG.info("Creating Affinity Matrix...Done!",LOG.ll.MATRIX)
        assert(W.shape == (X.shape[0],X.shape[0]))
        if np.max(W)==0:
            raise Exception("Affinity matrix cannot have all entries equal to zero.")
        
        if not hook is None:
            hook._end(X=X,Y=Y,W=W)

        return AffMat(W=W.astype(np.float32),cache_dir=self.cache_dir)

    
    def __init__(self,dist_func, mask_func, metric="euclidean",cache_dir = None,num_anchors=None, **arg):
        """ Constructs the Affinity Matrix Generator.
        
        Args:
            X (`NDArray[float].shape[N,D]`): A matrix containing the vertex positions. 
            dist_func (str): specifies the distance function to be used. Supported values:
                {
                    * gaussian: ``np.exp(-(d*d)/(2*sigma*sigma))``, where d is the distance. Requires ``sigma`` on **kwargs.
                    * LNP: Linear neighborhood propagaton. Requires ``k`` on **kwargs. 
                    * NLNP: Normalized reciprocal of  Linear neighborhood propagation. Requires ``k`` on **kwargs.
                    * constant: Every weight is set to 1.
                    * inv_norm: ``1/d``, where d is the distance
                }
            mask_func (str): specifies the function used to determine the neighborhood. Supported Values:
                {
                    * epsilon: Epsilon-neighborhood. Requires ``eps`` on **args.
                    * knn / mutKNN / symKNN: K-nearest neighbors Requires ``k`` on **args. Default knn is equal to symKNN
                    * load: loads CSR matrix specified by `load_path`
                }
            cache_dir(Union[str,None]): If not `None`, ignores everything and simply loads matrix from cache directory. See `tf_labelprop.gssl.graph.affmat_gen.Affmat` for reference...
            metric (str): specifies the metric when computing the distance. Default is `euclidean`. See the documentation of
                `scipy.spatial.distance.cdist` for more details.
            **arg: Remaining arguments.

        """
        self.K = None
        self.sigma = None
        self.metric = metric
        self.dist_func_str = dist_func
        self.cache_dir = cache_dir
        self.num_anchors = num_anchors
        
        if "row_normalize" in arg and arg["row_normalize"] == True:
            self.row_normalize = True
        else:
            self.row_normalize = False
            
        
        
        if dist_func in ["LNP","NLNP"]:
            mask_func = dist_func
        if mask_func in ["LNP","NLNP"]:
            dist_func = mask_func
            
        mask_func = mask_func.lower()
        if mask_func == 'knn':
            knn_mode = "sym"
        if mask_func in ["mutknn","symknn"]:
            knn_mode = mask_func[:-3]
            mask_func = 'knn'
        
        
        
        if dist_func == "gaussian":
            if not "sigma" in arg:
                raise ValueError("Did not specify sigma for gaussian")
            
            self.sigma = arg["sigma"]
            self.dist_func = partial(lambda d: np.exp(-(d*d)/(2*self.sigma*self.sigma)))
        elif dist_func == "constant":
            self.dist_func = lambda d: 1
        elif dist_func == "inv_norm":
            self.dist_func = lambda d: 1/(d+1e-09)
        
        if mask_func == "load":
            self.mask_func = "load"
        elif mask_func == "eps":
            if not "sigma" in arg:
                raise ValueError("Did not specify eps parameter for epsilon-neighborhood")
            self.mask_func = partial(lambda X,eps: epsilonMask(X, eps),eps=arg["eps"])
        elif mask_func == "knn":
            if not "k" in arg:
                raise ValueError("Did not specify k parameter for knn-neighborhood")
            self.mask_func = partial(lambda X,k,mode: knnMask(X, k, mode=mode),k=arg["k"],mode=knn_mode)
        elif mask_func == "LNP":
            if not "k" in arg:
                raise ValueError("Did not specify k for LNP")
            self.mask_func = partial(lambda X,K: LNP(X, K))
        elif mask_func == "NLNP":
            if not "k" in arg:
                raise ValueError("Did not specify k for NLNP")
            self.mask_func = partial(lambda X,K: NLNP(X,K))
        
        

def epsilonMask(X,eps,metric="euclidean"):
    """
    Calculates the distances only in the epsilon-neighborhood.
    
    Args:
        X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
        eps (float) : A parameter such that K[i,j] = 0 if dist(X_i,X_j) >= eps.
    Returns:
        `NDArray[int].shape[N,N]` : a dense matrix ´K´ of shape `[N,N]` whose nonzero 
        **[i,j]** entries correspond to distances between neighbors **X[i,:],X[j,:]** .
    
    """
    assert isinstance(X, np.ndarray)
    
    K = scipydist.cdist(X,X,metric=metric)
    rows,cols = np.where(K > eps)
    K[rows,cols] = 0
    return(K)



def __symmetrize_KNN(W,mode):
    
    if not mode in ['mut','sym','fsym']:
        raise ValueError("Unrecognized KNN mode {}".format(mode))
    
    if mode == 'mut':
        W = W.minimum(W.T)
    elif mode == 'sym':
        W = W.maximum(W.T)
    elif mode == 'fsym':
        W = 0.5*(W + W.T)
    return W

def knnMask(X,k,mode='sym',metric="euclidean"):
    """
    Calculates the distances only in the knn-neighborhood.
    
    Args:
        X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
        k (int) :  A parameter such that ´K[i,j] = 1´ iff X_i is one of the k-nearest neighbors of X_j
        mode (str) : type of KNN. Supported values:
                {
                    * ``mut``.``K[i,j] = min(K[i,j],K[j,i])``.
                    * ``sym``. ``K[i,j] = max(K[i,j],K[j,i])``.
                    * ``none``. No symmetrization. WARNING: Many GSSL algorithms depend on a symmetric affinity matrix.
                }
    Returns:
        `NDArray[int].shape[N,N]` : a dense matrix ´K´ of shape `[N,N]` whose nonzero 
        **[i,j]** entries correspond to distances between neighbors **X[i,:],X[j,:]** .
    
    """

    if X.shape[0] > 1000:
        K =  _faiss_knn(X, k, mode=mode)
        return K
        
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree',metric=metric).fit(X)
    K = np.zeros((X.shape[0],X.shape[0]))
    for i in np.arange(X.shape[0]):
        distances, indices = nbrs.kneighbors([X[i,]])
        
        for dist, index in zip(distances,indices):
            K[i,index] = np.array(dist)
    

    K = scipy.sparse.csr_matrix(K)
    K = __symmetrize_KNN(K,mode=mode)
    
    return K

def _faiss_knn(X,k, mode='mut', inner_prod = False):
    # kNN search for the graph
    X  = np.ascontiguousarray(X)
    print("Number of GPUS detected by FAISS: {}".format(faiss.get_num_gpus() ))
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0

    c = time.time()
    if inner_prod:
        faiss.normalize_L2(X)
        
        index =  faiss.GpuIndexFlatIP(res,d,flat_config)         
    else:
        index = faiss.GpuIndexFlatL2(res,d,flat_config)   # build the index
    #normalize_L2(X)
    elapsed = time.time() - c
    LOG.info(f'kNN Index built in {elapsed:.3f} seconds',LOG.ll.UTILS)
    index.add(X) 
    N = X.shape[0]
    Nidx = index.ntotal


    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    LOG.info(f'kNN Search done in {elapsed:.3f} seconds',LOG.ll.UTILS)



    # Create the graph
    D = np.sqrt(D[:,1:])
    
    
    
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    
    
    W =  __symmetrize_KNN(W,mode=mode)

    return W

def __quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    
    from quadprog import solve_qp    
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]   

def LNP(X,K, symm = True):
    """ Computes the edge weights through Linear Neighborhood Propagation.
    
    Args:
         X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
         K (`NDArray[float].shape[N,N]`) : Dense Affinity mask, whose positive entries correspond to neighbors.
    Returns:
        `NDArray[float].shape[N,N]` : A dense affinity matrix whose weights minimize the linear reconstruction of each instance.
        
    """
    W = np.zeros((X.shape[0],X.shape[0]))

    if K.shape[0] !=  X.shape[0]:
        raise ValueError("Incompatible shapes for X,K")
    
    
    if K.shape[0] == 0:
        return K
    
    P = {}
    q = {}
    G = {}
    h = {}
    A = {}
    b = {}
    
    num_nbors = np.zeros((K.shape[0]))
    all_indices = [None] * X.shape[0]
    
    for i in range(K.shape[0]):
        all_indices[i] = (np.where(W[i,] > 0))
        num_nbors[i] = str(all_indices[i].shape[0])
        k = num_nbors[i]
        if not k in P.keys():
            P = np.zeros([k,k])
        if not k in q.keys():
            q[k] = np.zeros((k))
        if not k in G.keys():
            G[k] = -np.identity(k)
        if not k in h.keys():
            h[k] = np.zeros((k))
        if not k in A.keys():
            A[k] = np.ones((1,k))
        if not k in b.keys():
            b[k] = np.ones((1))
        
    
    for i in np.arange(X.shape[0]):
        k = num_nbors[i]
        indices = all_indices[i]
        for m in range(k):
            for n in range(k):
                P[m,n] = np.dot((X[i,]-X[indices[m],]),(X[i,]-X[indices[n],]).T)        
        for m in range(k):
            P[m,m] += 1e-03
          
        W_lnp = __quadprog_solve_qp(P[k], q[k], G[k], h[k], A[k], b[k])
        
        for m in range(k):
            W[i,indices[m]] = W_lnp[m]
    if symm:
        W = 0.5*(W + W.T)
    
    return(W)

def NLNP(X,K, symm = True):
    """ Computes the normalized reciprocals of the edge weights through Linear Neighborhood Propagation.
    
    Args:
         X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
         K (`NDArray[float].shape[N,N]`) : Dense Affinity mask, whose positive entries correspond to neighbors.
    Returns:
        `NDArray[float].shape[N,N]` : A dense affinity matrix whose weights are the normalized reciprocals of the ones given by Linear Neighborhood Propagation.
            
    """
    W = LNP(X,K,symm)
    for i in range(W.shape[0]):
        nonz = (np.where(W[i,] > 0))
        W[i,nonz] = np.reciprocal(W[i,nonz])
        W[i,nonz] = W[i,nonz] / np.linalg.norm(W[i,nonz])
    return(W)



