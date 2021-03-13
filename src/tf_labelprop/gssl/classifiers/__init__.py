import numpy as np
from tf_labelprop.gssl.classifiers.clgc import CLGC_Classifier 
from tf_labelprop.gssl.classifiers.gfhf import GFHF
from tf_labelprop.gssl.classifiers.gtam import GTAMClassifier
from  tf_labelprop.gssl.classifiers.lgc import LGCClassifier
from tf_labelprop.gssl.classifiers.lgc_lvo_auto_d import LGC_LVO_AUTO_D 
from  tf_labelprop.gssl.classifiers.rf import RandomForest
from  tf_labelprop.gssl.classifiers.siis import SIISClassifier


class GSSLClassifier(object):
    """ Skeleton class for GSSL Classifiers. """

    def CLEAN_UNLABELED_ROWS(self,Y,labeledIndexes):
        """
            Returns a copy of Y as a 2D matrix, with the unlabeled rows set to zero
        """
        from tf_labelprop.gssl.graph.gssl_utils import init_matrix
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y = init_matrix(Y,labeledIndexes)
        Y[np.logical_not(labeledIndexes),:] = 0
        return Y
    
    
    @classmethod
    def autohooks(cls, fun):
        """ Automatically calls the begin and end method of the hook. The classifier result is passed as 
        the 'Y' argument at the end."""
        
        def wrapper(self, *args, **kwargs):
            dct = dict(zip(fun.__code__.co_varnames[1:(len(args)+1)],args))
            slf = self
            
            kwargs.update(dct)
            hook = kwargs["hook"]
            
            if not hook is None:
                hook._begin(**kwargs)         
            
            kwargs["self"] = slf
            F = fun(**kwargs)
            
            kwargs["Y"] = F
            if not hook is None:
                kwargs.pop("self")
                hook._end(**kwargs)   
            return F
        return wrapper
    
    
   
    @staticmethod
    def get_name(self):
        return "---"
        


    
    
    
    def fit (self,X,W,Y,labeledIndexes, hook=None):
        """ Classifies the input data.
        
        Args:
            X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
            W (`NDArray[float].shape[N,N]`): The affinity matrix encoding the weighted edges.
            Y (`NDArray[float].shape[N,C]`): The initial belief matrix
            hook (GSSLHook): Optional. A hook to execute extra operations (e.g. plots) during the algorithm
        
        Returns:
            `NDArray[float].shape[N,C]`: An updated belief matrix.
        """
        if not hook is None:
            hook._begin(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)
            Y = self.CLEAN_UNLABELED_ROWS(Y, labeledIndexes)
            hook._end(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)         
        return Y



#from tf_labelprop.gssl.classifiers.lgc_lvo_auto_backup import LGC_LVO_AUTO_D


if __name__ == "__main__":
    print(GSSLClassifier().get_name())
    print(LGCClassifier().get_name())


