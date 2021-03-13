'''
Created on 27 de mar de 2019

@author: klaus
'''
import os 

import numpy as np
import scipy.linalg as sp
import scipy.sparse.linalg as ssp
from tf_labelprop.gssl.classifiers import GSSLClassifier
import tf_labelprop.gssl.graph.gssl_utils as gutils
import tf_labelprop.logging.logger as LOG
from tf_labelprop.output.folders import RESULTS_FOLDER


class LGC_LVO_AUTO_D(GSSLClassifier):
    """ Manifold Regularization with Laplacian Eigenmaps. 
        Minimizes the Least Squares in a semi-supervised way by using a linear combination of the first :math:`p` eigenfunctions. See :cite:`belkin2003`.
    """
    
    def get_name(self):
        return "LGC_LVO_AUTO_D"

    @GSSLClassifier.autohooks
    def __MR(self,X,W,Y,labeledIndexes,p,optimize_labels,hook=None):
        """
            -------------------------------------------------------------
                INITIALIZATION
            --------------------------------------------------------------
        """
        
        ORACLE_Y = Y.copy()
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        l = np.reshape(np.array(np.where(labeledIndexes)),(-1))
        num_lab = l.shape[0]
        
        
        if not isinstance(p, int):
            p = int(p * num_lab)
    
        if p > Y.shape[0]:
            p = Y.shape[0]
            LOG.warn("Warning: p greater than the number of labeled indexes",LOG.ll.CLASSIFIER)
        #W = gutils.scipy_to_np(W)
        #W =  0.5* (W + W.T)
        L = gutils.lap_matrix(W,  which_lap='sym')
        D = gutils.deg_matrix(W,flat=True,pwr=-1.0)
        
        L = 0.5*(L+L.T)
        
        def check_symmetric(a, tol=1e-8):
            return np.allclose(a, a.T, atol=tol)
        def is_pos_sdef(x):
            return np.all(np.linalg.eigvals(x) >= -1e-06)
        import scipy.sparse
        sym_err = L - L.T
        sym_check_res = np.all(np.abs(sym_err.data) < 1e-7)  # tune this value
        assert sym_check_res
        
        """---------------------------------------------------------------------------------------------------
                EIGENFUNCTION EXTRACTION
        ---------------------------------------------------------------------------------------------------
        """
        import time
        start_time = time.time()
        
        import os.path as osp
        from tf_labelprop.settings import INPUT_FOLDER

        
        cache_eigvec = osp.join(INPUT_FOLDER,'eigenVectors.npy')
        cache_eigval = osp.join(INPUT_FOLDER,'eigenValues.npy')
        

        if False:
            eigenValues, eigenVectors = np.load(cache_eigval), np.load(cache_eigvec)
            eigenVectors = eigenVectors[:,:p]
            eigenValues = eigenValues[:p]
        else:
        
            eigenVectors, eigenValues = W.load_eigenfunctions(p)
            
            time_elapsed = time.time() - start_time
            LOG.info("Took {} seconds to calculate eigenvectors".format(int(time_elapsed)))
            idx = eigenValues.argsort() 
            eigenValues = eigenValues[idx]
            LOG.debug(eigenValues)
            assert eigenValues[0] <= eigenValues[eigenValues.shape[0]-1]
            eigenVectors = eigenVectors[:,idx]
            np.save(cache_eigval,arr=eigenValues)
            np.save(cache_eigvec,arr=eigenVectors)
        U = eigenVectors
        LAMBDA = eigenValues
        
        
        U = U[:,np.argsort(LAMBDA)]
        LAMBDA = LAMBDA[np.argsort(LAMBDA)]
        
        import tensorflow as tf
                
        gpus = tf.config.experimental.list_physical_devices('GPU')

        #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)])
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        """
        -------------------------------------------------------------------------
            Define Constants on GPU
        ------------------------------------------------------------------------------
        """
        U, X, Y = [tf.constant(x.astype(np.float32)) for x in [U,X,Y]]
        
        
        _U_times_U = tf.multiply(U,U)
        N = X.shape[0]
        

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
            x = tf.maximum(x,0*x)
            x = x + eps # [N,C]    [N,1]
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
        
        
        
        """
            -----------------------------------------------------------------------------
            DEFINE VARS
            --------------------------------------------------------------------------------
        """
        
        MU = tf.Variable(0.1,name="MU")
        
        LAMBDA = tf.constant(LAMBDA.astype(np.float32),name="LAMBDA")        
        PI = tf.Variable(tf.ones(shape=(tf.shape(Y)[0],),dtype=tf.float32),name="PI")
        _l = LAMBDA.numpy()
        CUTOFF = tf.Variable(0.0,name='CUTOFF')
        CUTOFF_K = tf.Variable(1.0)
        @tf.function
        def get_alpha(MU):
            return tf.pow(2.0,-tf.math.reciprocal(tf.abs(100*MU)))
        @tf.function
        def to_prob(x):
            return tf.nn.softmax(x,axis=1)
        @tf.function
        def cutoff(x):
            return 1.0/(1.0+tf.exp(-CUTOFF_K*(CUTOFF-x)))
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(8,kernel_size=5,padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(8,kernel_size=5,padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv1D(1,kernel_size=3,padding='same'))

        model.add(tf.keras.layers.Flatten())
        
        
        """
            -----------------------------------------------------------------------------
            DEFINE FORWARD
            --------------------------------------------------------------------------------
        """
        @tf.function
        def forward(Y,U,PI, mode='train',remove_diag=True):
            if mode == 'train':
                U = tf.gather(U,indices=np.where(labeledIndexes)[0],axis=0)
                Y = tf.gather(Y,indices=np.where(labeledIndexes)[0],axis=0)
                #F = tf.gather(F,indices=np.where(labeledIndexes)[0],axis=0)
                
                PI = tf.gather(PI,indices=np.where(labeledIndexes)[0],axis=0)
                
            
            pi_Y = spd_matmul(to_sp_diag(tf.abs(PI)),Y )
            
            alpha = get_alpha(MU)
            
            
            """
                Maybe apply custom convolution to LAMBDA, otherwise just fit LGC's alpha using the corresponding filter 1/(1-alpha + alpha*lambda)
            """
            if not self.custom_conv:
                lambda_tilde = tf.math.reciprocal(1-alpha + alpha*LAMBDA)
            else:
                #lambda_tilde = tf.math.reciprocal(1-alpha + alpha*LAMBDA)
                _lambda = (LAMBDA - tf.reduce_mean(LAMBDA)) / tf.math.reduce_std(LAMBDA)
                lambda_tilde = tf.clip_by_value(2*tf.nn.sigmoid(tf.reshape(model(_lambda[None,:,None]),(-1,))),0,1)
                lambda_tilde = tf.sort(lambda_tilde,direction='DESCENDING')
            lambda_tilde = tf.reshape(divide_by_row(lambda_tilde[None,:]),(-1,))



            _self_infl = mult_each_row_by(tf.square(U),by=lambda_tilde) #Square each element of U, then dot product of each row with lambda_tilde
            _self_infl = tf.reduce_sum(_self_infl,axis=1)
            
            _P_op = U @ (mult_each_col_by(  (tf.transpose(U) @  pi_Y) ,by=lambda_tilde )  )
            if not remove_diag :
                _diag_P_op = tf.zeros_like(mult_each_col_by(pi_Y,by=_self_infl))
            else:
                _diag_P_op = mult_each_col_by(pi_Y,by=_self_infl)
            return divide_by_row(_P_op-_diag_P_op), lambda_tilde, pi_Y
        
        """
            -----------------------------------------------------------------------------
                DEFINE LOSSES and learning schedule
            --------------------------------------------------------------------------------
        """
        losses = {
            'xent': lambda y_, y: tf.reduce_mean(-tf.reduce_sum(y_ * tf.cast(tf.math.log(smooth_labels(y,factor=0.01)),tf.float32),axis=[1])),
            'sq_loss' : lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.square(y_-y),axis=[1])),
            'abs_loss' : lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.abs(y_-y),axis=[1])),
            'hinge' : lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.maximum(1. - y_ * y,tf.zeros_like(y)),axis=1))
        }
            
        NUM_ITER = 700
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.5,
            decay_steps=200,
            decay_rate=0.9,
            staircase=False)

        opt = tf.keras.optimizers.Adam(0.05)
        
        Y_l = tf.gather(Y,indices=np.where(labeledIndexes)[0],axis=0)
        
        
        #import matplotlib.pyplot as plt
        #import matplotlib
        #matplotlib.use('tkagg')
        import pandas as pd
        """
            -----------------------------------------------------------------------------
            LEARNING
            --------------------------------------------------------------------------------
        """
        L = []
        df = pd.DataFrame()
        max_acc, min_loss = [0,np.inf]
        for i in range(NUM_ITER):
            #MU.assign(i)
            with tf.GradientTape() as t:
                # no need to watch a variable:
                # trainable variables are always watched
                pred_L, lambda_tilde, pi_Y = forward(Y,U,PI,mode='train')
                loss_sq = losses['sq_loss'](pred_L,Y_l)
                loss = losses['xent'](pred_L,Y_l) 
                
                loss_xent = losses['xent'](pred_L,Y_l)
                
            acc = accuracy(Y_l,pred_L)
            _not_lab = np.where(np.logical_not(labeledIndexes))[0]
            acc_true = accuracy(tf.gather(ORACLE_Y,indices=_not_lab,axis=0), 
                                tf.gather(forward(Y,U,PI,mode='eval')[0],indices=_not_lab,axis=0)
                                )
            
            L.append(np.array([i,loss_sq,loss,loss_xent,acc,acc_true])[None,:])
            
            
            """
                TRAINABLE VARIABLES GO HERE
            """
            if self.custom_conv:
                trainable_variables =  model.weights 
            else:
                trainable_variables = [MU]
            if optimize_labels:
                trainable_variables.append(PI)
            
                    
            
            if acc > max_acc:
                print(max_acc)
                best_trainable_variables =  [k.numpy() for k in trainable_variables]
                max_acc = acc
                min_loss = loss
                counter_since_best = 0
            elif acc <= max_acc:
                
                counter_since_best += 1
                if counter_since_best > 2000:
                    break
                    
            """
                Apply gradients
            """
            gradients = t.gradient(loss, trainable_variables)
            opt.apply_gradients(zip(gradients, trainable_variables))
            """
                Project labels such that they sum up to the original amount
            """
            pi = PI.numpy()
            pi[labeledIndexes] = np.sum(labeledIndexes) * pi[labeledIndexes]/(np.sum(pi[labeledIndexes]))
            PI.assign(pi) 
            
            if i % 10 == 0:
                """ Print info """
                if not hook is None:
                    if self.hook_iter_mode == "labeled":
                        plot_y = np.zeros_like(Y)
                        plot_y[labeledIndexes] = Y_l.numpy()
                    else:
                        plot_y = tf.clip_by_value(forward(Y,U,PI,mode='eval')[0],0,999999).numpy()
                    hook._step(step=i,X=X,W=W,Y=plot_y,labeledIndexes=labeledIndexes) 
                alpha = get_alpha(MU)
                PI_l = tf.gather(PI,indices=np.where(labeledIndexes)[0],axis=0)
                LOG.info(f"Acc: {acc.numpy():.3f}; ACC_TRUE:{acc_true.numpy():.3f}  Loss: {loss.numpy():.3f}; alpha = {alpha.numpy():.3f}; PI mean = {tf.reduce_mean(PI_l).numpy():.3f} ")
        
        
        #plt.scatter(range(lambda_tilde.shape[0]),np.log10(lambda_tilde/LAMBDA),s=2)
        #plt.show()
        for k in range(len(trainable_variables)):
            trainable_variables[k].assign(best_trainable_variables[k])
        return tf.clip_by_value(forward(Y,U,PI,mode='eval')[0],0,999999).numpy()
        
        
        

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        """
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
        print(np.argmax(Y[labeledIndexes,:],axis=1))
        clf.fit(X[labeledIndexes,:], np.argmax(Y[labeledIndexes,:],axis=1))
        F = clf.predict(X)
        from tf_labelprop.gssl.graph.gssl_utils import init_matrix as mat
        print(F[labeledIndexes])
        print(Y[labeledIndexes])
        return mat(F,[True]*F.shape[0])
        """
        return (self.__MR(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes,p=self.p,
                         optimize_labels=self.optimize_labels,hook=hook))


    def __init__(self,p, optimize_labels=False, custom_conv = True,hook_iter_mode="all"):
        """ Constructor for LGC_LVO_AUTO_D classifier.
            
        Args:
            p (Union[float,int]). The number of eigenfunctions. It is given as either the absolute value if integer, or a percentage of
                the labeled data. if float. Default is ``0.2``
            optimize_labels (bool). Whether to optimize for label reliability. Default is ``True``
        """
        self.p = p
        self.optimize_labels = optimize_labels
        self.custom_conv = custom_conv
        self.hook_iter_mode = hook_iter_mode