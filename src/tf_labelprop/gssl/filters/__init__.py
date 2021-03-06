class GSSLFilter(object):
    """ Skeleton class for GSSL Filters. """
    
        
    @staticmethod
    def get_name(self):
        return "---"

    
    @classmethod
    def autohooks(cls, fun):
        """ Automatically calls the begin and end method of the hook. At the end, the filtered labels are passed as 'Y',
        and the new labeled indexes as 'labeledIndexes'."""
        
        def wrapper(self, *args, **kwargs):
            dct = dict(zip(fun.__code__.co_varnames[1:(len(args)+1)],args))
            slf = self
            
            kwargs.update(dct)
            hook = kwargs["hook"]
            
            if not hook is None:
                hook._begin(**kwargs)         
            
            kwargs["self"] = slf
            F, lb = fun(**kwargs)
            
            kwargs["Y"] = F
            kwargs["labeledIndexes"] = lb
            if not hook is None:
                kwargs.pop("self")
                hook._end(**kwargs)   
            return F,lb
        return wrapper
    
    
    def fit (self,X,Y,labeledIndexes,W = None,hook=None):
        """ Filters the input data.
        
        Args:
            X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
            Y (`NDArray[float].shape[N,C]`): A (noisy) belief matrix
            labeledIndexes(`NDArray[bool].shape[N]`)  : Indices to be marked as labeled.
            W (`NDArray[float].shape[N,N]`): Optional. The affinity matrix encoding the weighted edges.
            hook (GSSLHook): Optional. A hook to execute extra operations (e.g. plots) during the algorithm
        
        Returns:
            `NDArray[float].shape[N,C]`: A corrected version of the belief matrix.
            `NDArray[bool].shape[N]`: Updated labeledIndexes.
        """
        if not hook is None:
            hook._begin(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)
            hook._end(X=X,W=W,Y=Y,labeledIndexes=labeledIndexes)  
        return Y, labeledIndexes
    
from tf_labelprop.gssl.filters.ldst_filter import LDST
from tf_labelprop.gssl.filters.ldst_removal import LDSTRemover
from tf_labelprop.gssl.filters.lgc_lvo import LGC_LVO_Filter
from tf_labelprop.gssl.filters.lgc_lvo_auto_l import LGC_LVO_AUTO_Filter
from tf_labelprop.gssl.filters.mr_removal import MRRemover


