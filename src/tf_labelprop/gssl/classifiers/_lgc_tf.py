'''
Created on 23 de nov de 2019

@author: klaus
'''
import time

from scipy import sparse

import numpy as np
import tensorflow as tf
from tf_labelprop.gssl.classifiers import GSSLClassifier
import tf_labelprop.logging.logger as LOG 


def get_S_fromtensor(W):
    wsum = tf.sparse.reduce_sum(W,axis=1)
    wsum = tf.reshape(wsum,(-1,))
    d_sqrt = tf.math.reciprocal(tf.sqrt(wsum))
    d_sqrt = tf.where(tf.math.is_finite(d_sqrt),d_sqrt,tf.ones(shape=tf.shape(d_sqrt)))
    
    d_sqrt_i = tf.gather(d_sqrt,W.indices[:,0])
    d_sqrt_j = tf.gather(d_sqrt,W.indices[:,1])
    
    
    S = tf.sparse.SparseTensor(indices=W.indices,
                                    values=W.values * d_sqrt_i * d_sqrt_j,
                                    dense_shape=W._dense_shape)
    
    return S

def update_F(TOTAL_ITER,ALPHA,S,F_0):
    i = tf.constant(0)
    c = lambda i,F: tf.less(i, TOTAL_ITER)
    b = lambda i,F: (tf.add(i, 1),(1 - ALPHA)*F_0 + ALPHA*tf.sparse.matmul(S,F))
    r = tf.while_loop(c, b, [i,F_0])
    return r
            
def get_P(TOTAL_ITER,ALPHA,S,F_0):
    i = tf.constant(0)
    c = lambda i,F: tf.less(i, TOTAL_ITER)
    b = lambda i,F: (tf.add(i, 1),(1 - ALPHA)*F_0 + ALPHA*tf.sparse.matmul(S,F))
    r = tf.while_loop(c, b, [i,F_0])
    return r



""" UTIL FUNCTIONS BEGIN """
def gather(x,F):
    with tf.name_scope("gather"):
        return tf.gather(F,tf.reshape(x,(-1,)))  
    

def repeat(x,n):
    with tf.name_scope('repeat'):
            x = tf.expand_dims(x, axis=-1)
            x = tf.tile(x,[1,n])
            return x
def row_normalize(x):
    with tf.name_scope('row_norm'):
        x = tf.clip_by_value(x,0.,1.)
        s= tf.cast(tf.shape(x)[1],tf.float32)
        vec = tf.reduce_sum(x,axis=1)
        x = x/repeat(vec,s)
        x = tf.where(tf.math.is_finite(x),x,tf.ones(shape=tf.shape(x))/s)
        return x
""" UTIL FUNTIONS END """

def convert_sparse_matrix_to_sparse_tensor(X,var_values=False):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    indices = np.reshape(np.asarray(indices),(-1,2))
    
    return tf.SparseTensor(indices, np.reshape(np.asarray(coo.data).astype(np.float32),(-1,)), coo.shape)
    

def LGC_iter_TF(X,W,Y,labeledIndexes, alpha = 0.1,num_iter = 1000, hook=None):
    c = time.time()
    
    """ Set W to sparse if necessary, make copy of Y """
    W = sparse.csr_matrix(W)        
    Y = np.copy(Y)
    
    """ Convert W to tensor """
    W = convert_sparse_matrix_to_sparse_tensor(W)
    LOG.debug(W,LOG.ll.CLASSIFIER)
    
    """ Get degree Matrix """
    D =  tf.sparse.reduce_sum(W,axis=1)
    
    
    """ F_0 is a copy of the label matrix, but we erase the information on labeled Indexes """
    F_0 = np.copy(Y).astype(np.float32) 
    F_0[np.logical_not(labeledIndexes),:] = 0.0
    
    
    
    """
        CREATE S - Needed for LGC propagation
    """
    S =  get_S_fromtensor(W)
    
    
    """
    CREATE F variable
    """
    F = tf.Variable(np.copy(F_0).astype(np.float32),name="F")
    F_0 = tf.Variable(F_0)
    TOTAL_ITER = tf.constant(int(num_iter))
    for _ in range(num_iter):
        F = (1-alpha)*F_0 + alpha*tf.sparse.sparse_dense_matmul(S,F)
    
    elapsed = time.time() - c
    LOG.info('Label Prop done in {:.2} seconds'.format(elapsed),
             LOG.ll.CLASSIFIER)
    
    return F.numpy()


