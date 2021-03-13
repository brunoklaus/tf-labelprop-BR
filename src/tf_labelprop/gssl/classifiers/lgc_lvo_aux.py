'''
Created on 19 de fev de 2021

@author: klaus
'''
import tensorflow as tf


def to_sp_diag(x):
    n = tf.cast(x.shape[0],tf.int64)
    indices = tf.concat([tf.range(n,dtype=tf.int64)[None,:],
                         tf.range(n,dtype=tf.int64)[None,:]],axis=0)
    return tf.sparse.SparseTensor(indices=tf.transpose(indices),values=x,dense_shape=[n,n])
        
@tf.function
def smooth_labels(labels, factor=0.001):
    # smooth the labels
    labels = tf.cast(labels,tf.float32)
    labels *= (1 - factor)
    labels += (factor / tf.cast(tf.shape(labels)[0],tf.float32))
    # returned the smoothed labels
    return labels
@tf.function
def divide_by_row(x,eps=1e-07):
    x = x - tf.tile(tf.reduce_min(x,axis=1)[:,None],(1,x.shape[1]))
    #x = tf.maximum(x,0*x+eps)
    #x = tf.abs(x) + eps
    #x = tf.abs(x)
    x = x + eps
    #x = x + eps # [N,C]    [N,1]
    return x / (tf.reduce_sum(x,axis=-1)[:,None])

def spd_matmul(x,y):
    return tf.sparse.sparse_dense_matmul(x,y)

def mult_each_row_by(X,by):
    """ Elementwise multiplies each row by a given row vector.
    
        For a 2D tensor, also correponds to multiplying each column by the respective scalar in the given row vector
        
        Args:
            X (Tensor)  
            by (Tensor[shape=(N,)]): row vector
    
    """
    #[N,C]  [N,1]
    return X * by[None,:]

def mult_each_col_by(X,by):
    #[N,C]  [1,C]
    return X * by[:,None]


@tf.function
def accuracy(y_true,y_pred):
    acc = tf.cast(tf.equal(tf.argmax(y_true,axis=-1),tf.argmax(y_pred,axis=-1)),tf.float32)
    acc = tf.cast(acc,tf.float32)
    return tf.reduce_mean(acc)