'''
Created on 27 de mar de 2019

@author: klaus
'''
import os 

import numpy as np
import pandas as pd
import scipy.linalg as sp
import scipy.sparse.linalg as ssp
import tensorflow as tf
from tf_labelprop.gssl.classifiers import GSSLClassifier
from tf_labelprop.gssl.classifiers.lgc_lvo_aux import mult_each_col_by
import tf_labelprop.gssl.graph.gssl_utils as gutils
import tf_labelprop.logging.logger as LOG
from tf_labelprop.output.folders import RESULTS_FOLDER


class LGC_LVO_AUTO_D(GSSLClassifier):
    """ Manifold Regularization with Laplacian Eigenmaps. 
        Minimizes the Least Squares in a semi-supervised way by using a linear combination of the first :math:`p` eigenfunctions. See :cite:`belkin2003`.
    """
    @staticmethod
    def get_name(self):
        return "LGC_LVO_AUTO_D"
    
    DEBUG = False
    @tf.function
    def get_alpha(self,MU):
        return tf.pow(1.5,-tf.math.reciprocal(tf.abs(MU)))    
    
    def create_3d_mesh(self,df):
        import tf_labelprop.gssl.classifiers.lgc_lvo_aux as aux
        # library
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import matplotlib
        matplotlib.use('tkagg')
        df = df.astype(np.float32)
        df['alpha'] = [self.get_alpha(x).numpy() for x in df['i'].values]
        
        df['acc_diff'] = df['acc_true'] - df['acc'] 
        
        df = df[df['p'] >= 1]
        for Z in ['prop','loss_sq','acc','acc_true','acc_diff']:
            df['X'], df['Y'], df['Z'] = df['i'], df['p'], df[Z]
            m = np.argmax(df[Z].values)
            print(f" Max: {df[Z].values[m]} at ({df['X'].values[m]},{df['Y'].values[m]})")
            m = np.argmin(df[Z].values)
            print(f" Min: {df[Z].values[m]} at ({df['X'].values[m]},{df['Y'].values[m]})")
            # Make the plot
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # to Add a color bar which maps values to colors.
            surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.get_cmap('viridis'), linewidth=0.2)
            fig.colorbar( surf, shrink=0.5, aspect=5)
            plt.title(Z)
            ax.set_xlabel('eigfuncs',rotation=150)
            ax.set_ylabel('i')
            ax.set_zlabel(Z,  rotation=60)
            plt.show()
            print(pd.unique(df['alpha']))
    
    
    


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
        L = gutils.lap_matrix(W)
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
        eigenVectors, eigenValues = W.load_eigenfunctions(p)
        
        time_elapsed = time.time() - start_time
        LOG.info("Took {} seconds to calculate eigenvectors".format(int(time_elapsed)))
        U = eigenVectors
        LAMBDA = eigenValues
        
        """
        -------------------------------------------------------------------------
            Import and setup Tensorflow
        ------------------------------------------------------------------------------
        """
        import tensorflow as tf
        import tf_labelprop.gssl.classifiers.lgc_lvo_aux as aux
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
        
        
        
        """
            -----------------------------------------------------------------------------
            DEFINE VARS
            --------------------------------------------------------------------------------
        """
        MU = tf.Variable(0.1,name="MU")
        
        LAMBDA = tf.constant(LAMBDA.astype(np.float32),name="LAMBDA")        
        PI = tf.Variable(tf.ones(shape=(tf.shape(Y)[0],),dtype=tf.float32),name="PI")
        _l = LAMBDA.numpy()
        
        
        """
            -----------------------------------------------------------------------------
            DEFINE FORWARD
            --------------------------------------------------------------------------------
        """
        def forward(Y,U,PI, mode='train',p=None,remove_diag=True):
            if p is None:
                p = 99999
            
            pi_Y = aux.spd_matmul(aux.to_sp_diag(tf.abs(PI)),Y )
            
            alpha = self.get_alpha(MU)
            
            
            """
                Maybe apply custom convolution to LAMBDA, otherwise just fit LGC's alpha using the corresponding filter 1/(1-alpha + alpha*lambda)
            """
            #tf.print(alpha)
            a = alpha - alpha * LAMBDA
            lambda_tilde = 1/(1-a)
            
            """ Set entries corresponding to eigvector e_i to zero for i > p """
            lambda_tilde = tf.where(tf.less_equal(tf.range(0,lambda_tilde.shape[0]),p),lambda_tilde,0*lambda_tilde)



            _self_infl = aux.mult_each_row_by(tf.square(U),by=lambda_tilde) #Square each element of U, then dot product of each row with lambda_tilde
            B = _self_infl
            _self_infl = tf.reduce_sum(_self_infl,axis=1)
            
            A = aux.mult_each_col_by(  (tf.transpose(U) @  pi_Y) ,by=lambda_tilde ) 
            _P_op = U @ (A)
            if not remove_diag :
                _diag_P_op = tf.zeros_like(aux.mult_each_col_by(pi_Y,by=_self_infl))
            else:
                _diag_P_op = aux.mult_each_col_by(pi_Y,by=_self_infl)
                
            if mode == 'eval':
                return aux.divide_by_row(_P_op - _diag_P_op)
            else:
                return A,B,    aux.divide_by_row(_P_op - _diag_P_op)
        
        def forward_eval(Y,U,PI, mode='train',p=None,remove_diag=True):
            if p is None:
                p = 99999
            
            pi_Y = aux.spd_matmul(aux.to_sp_diag(tf.abs(PI)),Y )
            
            alpha = self.get_alpha(MU)
            
            
            """
                Maybe apply custom convolution to LAMBDA, otherwise just fit LGC's alpha using the corresponding filter 1/(1-alpha + alpha*lambda)
            """
            #tf.print(alpha)
            a = alpha - alpha * LAMBDA
            lambda_tilde = 1/(1-a)
            
            """ Set entries corresponding to eigvector e_i to zero for i > p """
            lambda_tilde = tf.where(tf.less_equal(tf.range(0,lambda_tilde.shape[0]),p),lambda_tilde,0*lambda_tilde)



            _self_infl = aux.mult_each_row_by(tf.square(U),by=lambda_tilde) #Square each element of U, then dot product of each row with lambda_tilde
            _self_infl = tf.reduce_sum(_self_infl,axis=1)
            
            A = aux.mult_each_col_by(  (tf.transpose(U) @  pi_Y) ,by=lambda_tilde ) 
            _P_op = U @ (A)
            if not remove_diag :
                _diag_P_op = tf.zeros_like(aux.mult_each_col_by(pi_Y,by=_self_infl))
            else:
                _diag_P_op = aux.mult_each_col_by(pi_Y,by=_self_infl)
                
            
            return aux.divide_by_row(_P_op - _diag_P_op)
        
        
        """
            -----------------------------------------------------------------------------
                DEFINE LOSSES and learning schedule
            --------------------------------------------------------------------------------
        """
        losses = {
            'xent': lambda y_, y: tf.reduce_mean(-tf.reduce_sum(y_ * tf.cast(tf.math.log(aux.smooth_labels(y,factor=0.01)),tf.float32),axis=[1])),
            'sq_loss' : lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.square(y_-y),axis=[1])),
            'abs_loss' : lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.abs(y_-y),axis=[1])),
            'hinge' : lambda y_, y: tf.reduce_mean(tf.reduce_sum(tf.maximum(1. - y_ * y,tf.zeros_like(y)),axis=1))
        }
            
        NUM_ITER = 10
        Y_l = tf.gather(Y,indices=np.where(labeledIndexes)[0],axis=0)
        U_l = tf.gather(U,indices=np.where(labeledIndexes)[0],axis=0)
        PI_l = tf.gather(PI,indices=np.where(labeledIndexes)[0],axis=0)

        """
            -----------------------------------------------------------------------------
            LEARNING
            --------------------------------------------------------------------------------
        """
        L = []
        df = pd.DataFrame()
        max_acc, min_loss = [0,np.inf]
        best_p = np.inf
        for i in range(NUM_ITER,0,-1):
            MU.assign(i)
            
            A,B, _ = forward(Y_l,U_l,PI_l,mode='train')
            
            
            a1 = np.zeros_like(Y_l)
            a2 = np.zeros_like(Y_l)
            
            for i1 in range(p):
                a2 += mult_each_col_by(X=Y_l,by=B[:,i1])
                a1 +=  mult_each_col_by(np.tile(A[i1,:][None,:],[a1.shape[0],1]),
                                         U_l[:,i1])
                
                
                
                pred_L = aux.divide_by_row(a1-a2)
                        
                
                loss_sq = losses['sq_loss'](pred_L,Y_l)
                loss = losses['xent'](pred_L,Y_l) 
                
                loss_xent = losses['xent'](pred_L,Y_l)
                
                acc = aux.accuracy(Y_l,pred_L)
                _not_lab = np.where(np.logical_not(labeledIndexes))[0]
                
                if self.DEBUG:
                    acc_true = aux.accuracy(tf.gather(ORACLE_Y,indices=_not_lab,axis=0), 
                                        tf.gather(forward_eval(Y,U,PI,mode='eval',p=i1),indices=_not_lab,axis=0)
                                        )
                    prop = np.max(pd.value_counts(tf.argmax(pred_L,1).numpy(),normalize=True).values)
                else:
                    acc_true = 0
                    prop = 0
                
                L.append(np.array([i,i1,loss_sq,loss,loss_xent,acc,acc_true,prop])[None,:])
                if  (max_acc < acc) or (acc == max_acc and min_loss > loss):
                    print(f"acc: {acc},p:{i1},Mu:{int(MU.numpy())}alpha:{self.get_alpha(MU.numpy()).numpy()}")
                    best_p = int(i1)
                    best_MU = int(MU.numpy())
                    max_acc = acc
                    min_loss = loss.numpy()
                    """
                    if self.DEBUG:
                        alpha = self.get_alpha(MU)
                        I = np.identity(Y.shape[0], dtype = np.float32)
                        match_true = tf.gather(np.linalg.inv(I- alpha*(I - gutils.lap_matrix(W,'sym')))@Y,_not_lab,axis=0)
                        F = forward_eval(Y,U,PI,mode='eval',p=best_p)
                        
                        match_approx = tf.gather(F,indices=_not_lab,axis=0)
                        match = aux.accuracy(match_true, match_approx)
                        
                        print(f"Match rate {np.round(100*match,3)} ")
                        print(f"LGC_acc = {np.round(100*aux.accuracy(match_true,tf.gather(ORACLE_Y,indices=_not_lab,axis=0)),3)} ")
                        print(f"LGCLVO_acc = {np.round(100*aux.accuracy(match_approx,tf.gather(ORACLE_Y,indices=_not_lab,axis=0)),3)} ")
                    """
            
            if i % 1 == 0:
                """ Print info """
                if not hook is None:
                    if self.hook_iter_mode == "labeled":
                        plot_y = np.zeros_like(Y)
                        plot_y[labeledIndexes] = Y_l.numpy()
                    else:
                        MU.assign(best_MU)
                        plot_y = tf.clip_by_value(forward(Y,U,PI,p=best_p,mode='eval'),0,999999).numpy()

                    hook._step(step=i,X=X,W=W,Y=plot_y,labeledIndexes=labeledIndexes) 
                alpha = self.get_alpha(MU)
                
                LOG.info(f"Acc: {max_acc.numpy():.3f};  Loss: {loss.numpy():.3f}; alpha = {alpha.numpy():.3f};")
        
        
        if self.DEBUG:
            df = pd.DataFrame(np.concatenate(L,axis=0),index=range(len(L)),columns=['i','p','loss_sq','loss','loss_xent','acc','acc_true','prop'])
            self.create_3d_mesh(df)
        
        
        
        print(f"BEst mu: {best_MU}; best p: {best_p}")
        MU.assign(best_MU)
        print(MU)

        
        return forward_eval(Y,U,PI,mode='eval',p=None).numpy()
        """
        ----------------------------------------------------
            PART 2
        -------------------------------------------------
        
        
        """
        
        opt = tf.keras.optimizers.Adam(0.05)
        
        max_acc = 0
        for i in range(7000):
            #MU.assign(i)
            with tf.GradientTape() as t:
                _,_, pred_L= forward(Y_l,U_l,tf.gather(PI,indices=np.where(labeledIndexes)[0],axis=0),mode='train',p=best_p)
                loss_sq = losses['sq_loss'](pred_L,Y_l)
                loss = losses['xent'](pred_L,Y_l) 
                
                loss_xent = losses['xent'](pred_L,Y_l)
                
            acc = aux.accuracy(Y_l,pred_L)
            _not_lab = np.where(np.logical_not(labeledIndexes))[0]
            acc_true = aux.accuracy(tf.gather(ORACLE_Y,indices=_not_lab,axis=0), 
                                tf.gather(forward(Y,U,PI,mode='eval')[0],indices=_not_lab,axis=0)
                                )
            
            L.append(np.array([i,loss_sq,loss,loss_xent,acc,acc_true])[None,:])
                                
            """
                Project labels such that they sum up to the original amount
            """
            pi = PI.numpy()
            pi[labeledIndexes] = np.sum(labeledIndexes) * pi[labeledIndexes]/(np.sum(pi[labeledIndexes]))
            PI.assign(pi) 
            
            
            """
                TRAINABLE VARIABLES GO HERE
            """
            trainable_variables = []
            if optimize_labels:
                trainable_variables.append(PI)
                                
            """
                Apply gradients
            """
            gradients = t.gradient(loss, trainable_variables)
            opt.apply_gradients(zip(gradients, trainable_variables))
                    
            
            if acc > max_acc:
                print(max_acc)
                best_trainable_variables =  [k.numpy() for k in trainable_variables]
                max_acc = acc
                min_loss = loss
                counter_since_best = 0
        
        
        for k in range(len(trainable_variables)):
            trainable_variables[k].assign(best_trainable_variables[k])
            
        
        return forward(Y,U,PI,mode='eval',p=None).numpy()
        
        """
        
        for c in df.columns:
            if c.startswith('loss'):
                df[c] = (df[c] - df[c].min())/(df[c].max()-df[c].min())
        
        for c in df.columns:
            if not c in 'i':
                plt.plot(df['i'],df[c],label=c)
        plt.legend()
        plt.show()
        
        #plt.scatter(range(lambda_tilde.shape[0]),np.log10(lambda_tilde/LAMBDA),s=2)
        #plt.show()
        """
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