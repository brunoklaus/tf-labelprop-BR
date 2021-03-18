'''
Created on 27 de mar de 2019

@author: klaus
'''
import datetime
import os

import scipy.sparse

import numpy as np
from tf_labelprop.gssl.classifiers import GSSLClassifier
import tf_labelprop.gssl.graph.gssl_utils as gutils
import tf_labelprop.logging.logger as LOG


class GTAMClassifier(GSSLClassifier):
    """ Classifier using Graph Transduction Through Alternating Minimization (GTAMClassifier - see :cite:`Wang2008`).
    
    """
    
    @staticmethod
    def get_name(self):
        return "GTAM"

    @GSSLClassifier.autohooks
    def __GTAM(self,X,W,Y,labeledIndexes,mu = 99.0,useEstimatedFreq=True,num_iter = None,
             constant_prop=False,hook=None):
        '''BEGIN initialization'''
        Y = self.CLEAN_UNLABELED_ROWS(Y, labeledIndexes)
        labeledIndexes = np.array(labeledIndexes)

        
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        
        
        """ Estimate frequency of classes"""
        if isinstance(useEstimatedFreq,bool):
            if useEstimatedFreq == False:
                estimatedFreq = np.repeat(1/num_classes,num_classes)
            elif useEstimatedFreq == True:
                estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
        LOG.debug("Estimated frequency: {}".format(estimatedFreq),LOG.ll.CLASSIFIER)

        
        
        
        D = gutils.deg_matrix(W, flat=True)
        #Identity matrix
        I = np.identity(W.shape[0])
        #Get graph laplacian
        L = gutils.lap_matrix(W, which_lap='sym')
        #Propagation matrix
        from scipy.linalg import inv as invert
        P = invert( I- 1/(1+mu) *(I-L) )*mu/(1+mu)
        
        P_t = P.transpose()
        #Matrix A
        A = ((P_t @ L) @ P) + mu* ((P_t - I) @ (P - I))
        A = 0.5*(A + A.transpose())
        
        if not hook is None:
            W = scipy.sparse.coo_matrix(W)
        
        Z = []
        Q = None
        
        
        #Determine nontuning iter
        if num_iter is None:
            num_iter = num_unlabeled
        else:
            num_iter = min(num_iter,num_unlabeled)
            
        id_min_line, id_min_col = -1,-1
        '''END initialization'''
        #######################################################################################
        '''BEGIN iterations'''
        for i in np.arange(num_iter):

            '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                Then, we normalize each row so that row sums to its estimated influence
            '''
            ul = np.logical_not(labeledIndexes)
            
            Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq,weigh_by_degree=True)


            if Q is None:
                #Compute graph gradient
                Q = np.matmul(A,Z)
                if not hook is None:
                    Q_pure = np.copy(Q)
                
                Q[labeledIndexes,:] = np.inf
                
            else:
                Q[id_min_line,:] = np.inf
                d_sj = np.sum(Z[labeledIndexes,id_min_col])
                d_sj1 = d_sj + Z[id_min_line,id_min_col]
                Q[ul,id_min_col] =\
                 (d_sj/(d_sj1) * Q[ul,id_min_col]) + (Z[id_min_line,id_min_col]/d_sj1 * A[ul,id_min_line])
            
            #Find minimum unlabeled index
            
            if constant_prop:
                    expectedNumLabels = estimatedFreq * sum(labeledIndexes)
                    actualNumLabels = np.sum(Y[labeledIndexes],axis=0)
                    class_to_label = np.argmax(expectedNumLabels-actualNumLabels)
                    id_min_col = class_to_label
                    id_min_line = np.argmin(Q[:,class_to_label])
                
                    
            else:
                id_min = np.argmin(Q)
                id_min_line = id_min // num_classes
                id_min_col = id_min % num_classes
            
                
            
            #Update Y and labeledIndexes
            labeledIndexes[id_min_line] = True
            Y[id_min_line,id_min_col] = 1
            
            
            
            #Maybe plot current iteration
            
            
            if not hook is None:
                hook._step(step=i,Y=Y,labeledIndexes=labeledIndexes,P=P,Z=Z,Q=Q_pure,
                           id_min_line=id_min_line,id_min_col=id_min_col)
        '''END iterations'''    
        ######################################################################################################
        if self.return_labels:
            return np.asarray(Z)
        else:
            return np.asarray(P@Z)
        return np.asarray(P@Z)
    
    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__GTAM(X,W,Y,labeledIndexes,
                           mu=self.mu,
                           useEstimatedFreq=self.useEstimatedFreq,
                           num_iter=self.num_iter,
                           constant_prop = self.constantProp,
                           hook = hook
                           ))


    def __init__(self, mu = 99.0,num_iter=None,use_estimated_freq=True,constantProp=False,know_true_freq=True,weigh_by_degree=False,
                 return_labels=False):
        """" Constructor for GTAMClassifier classifier.
        
        Args:
            mu (float) :  a parameter determining the importance of the fitting term. Default is ``99.0``.
            num_iter (int) : Optional. The number of iterations to run. The default behaviour makes it N iterations given
                a NDArray[float].shape[N,D] input matrix.
            useEstimatedFreq (Union[bool,NDArray[C],None]) : If ``True``, then use estimated class freq. to balance the propagation.
                If it is a float array, it uses that as the frequency. If ``None``, assumes classes are equiprobable. Default is ``True``.
            useConstantProp (bool) : If ``True``, then use try to maintain a constant proportion of labels
                in all iterations.
                    
            
        """
        self.return_labels = return_labels
        self.mu = mu
        self.num_iter = num_iter
        self.useEstimatedFreq = use_estimated_freq
        self.constantProp = constantProp
        self.know_true_freq = know_true_freq
        self.weigh_by_degree = weigh_by_degree
        