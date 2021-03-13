
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tf_labelprop.gssl.classifiers import GSSLClassifier
from tf_labelprop.gssl.graph.gssl_utils import init_matrix
import tf_labelprop.gssl.graph.gssl_utils as gutils


class RandomForest(GSSLClassifier):
    """ Supervised Random Forest classifier
    """
    @staticmethod
    def get_name(self):
        return "RF"
    
    @GSSLClassifier.autohooks
    def __RF(self,X,W,Y,labeledIndexes,n_estimators, hook=None):
        rf = RandomForestClassifier(n_estimators=n_estimators,verbose=2)
        rf.fit(X[labeledIndexes,:],np.argmax(Y[labeledIndexes,:],axis=1) )
        pred = rf.predict(X)
        
        return init_matrix(pred, np.ones(X.shape[0],).astype(np.bool))   
    

    def fit (self,X,W,Y,labeledIndexes, hook=None):
        return(self.__RF(X,W, Y, labeledIndexes, self.n_estimators, hook))
    
    
    def __init__(self,n_estimators=10):
        """ Constructor for the LGC classifier.
            
            Args:
                alpha (float): A value between 0 and 1 (not inclusive) for alpha.
        """
        self.n_estimators = n_estimators
