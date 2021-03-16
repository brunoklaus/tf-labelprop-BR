'''
Created on 27 de mar de 2019

@author: klaus
'''
import numpy as np
from tf_labelprop.experiment.prefixes import *
from tf_labelprop.experiment.selector import Hook, select_and_add_hook
from tf_labelprop.experiment.selector import select_input, select_affmat, select_classifier, select_noise, \
    select_filter
from tf_labelprop.gssl.filters import LGC_LVO_AUTO_Filter
from tf_labelprop.gssl.graph.gssl_affmat import AffMatGenerator
import tf_labelprop.gssl.graph.gssl_utils as gutils
from tf_labelprop.input.dataset._toy_ds import ChapelleDataset
import tf_labelprop.logging.logger as LOG


## The hooks being utilized
PLOT_HOOKS = [Hook.INIT_LABELED,Hook.INIT_ALL,Hook.NOISE_AFTER,Hook.ALG_RESULT,Hook.ALG_ITER] \
                #+ [Hook.GTAM_Q,Hook.GTAM_F,Hook.GTAM_Y]
W_PLOT_HOOKS = [Hook.W_INIT_LABELED,Hook.W_INIT_ALL,Hook.W_NOISE_AFTER,Hook.FILTER_ITER,Hook.W_FILTER_AFTER,Hook.ALG_RESULT,Hook.ALG_ITER] \
                #+ [Hook.GTAM_Q,Hook.GTAM_F,Hook.GTAM_Y]

W_PLOT_HOOKS_NOITER = list(W_PLOT_HOOKS)
W_PLOT_HOOKS_NOITER.remove(Hook.ALG_ITER)
W_PLOT_HOOKS_NOITER.remove(Hook.FILTER_ITER)

PLOT_HOOKS_NOITER = list(PLOT_HOOKS)
PLOT_HOOKS_NOITER.remove(Hook.ALG_ITER)

TIME_HOOKS = [Hook.T_ALG,Hook.T_FILTER,Hook.T_NOISE,Hook.T_AFFMAT]                



def keys_multiplex(args):
    _Z =[("AFFMAT",AFFMAT_PREFIX),
                ("INPUT",INPUT_PREFIX),
                ("FILTER",FILTER_PREFIX),
                ("NOISE",NOISE_PREFIX),
                ("ALG",ALG_PREFIX),
                ("GENERAL",GENERAL_PREFIX)\
                ]

    mplex = {}
    for x,y in _Z:
        mplex[x] = {}
    
    for k,v in args.items():
        for x,y in _Z:
            if k.startswith(y):
                mplex[x][k[len(y):]] = v
                break
    return mplex



def postprocess(mplex):
    """ Performs some postprocessing on the multiplexed keys. """
    mplex = dict.copy(mplex)
    
    id = mplex["GENERAL"]["id"]
    for k in mplex.keys():
        if k == "ALG" or k == "FILTER":
            continue
        mplex[k]["seed"] = id
        
    if "tuning_iter_as_pct" in mplex["FILTER"].keys() and mplex["FILTER"]["tuning_iter_as_pct"]:
        mplex["FILTER"]["tuning_iter"] = mplex["NOISE"]["corruption_level"] *\
                                         mplex["FILTER"]["tuning_iter"]
        mplex["FILTER"].pop("tuning_iter_as_pct")
                                         
        
    
    return mplex

class ExperimentRun():
    """ Encapsulates a single run which derives from a  experiment, composed of the following steps.
         
         1. Reading the input features and true labels.
         2. Apply some noise process to the true labels, obtaining the corrupted labels.
         3. Create the Affinity matrix from the input features (and, optionally, noisy labels).
         4. Apply some filter to the corrupted labels, obtaining filtered labels.
         5. Run an GSSL algorithm to obtain the classification
         6. Get performance measures from the classification and filtered labels.
         
         
     Attributes:
        X (NDArray[float].shape[N,D]): The calculated input matrix
        W (NDArray[float].shape[N,N]): The affinity matrix encoding the graph.
        Y (NDArray[float].shape[N,C]): Initial label matrix
    """
    
    

    
    def __init__(self,args):
        self.args = dict(args)
        self.X = None
        self.labeledIndexes = None
        self.labeledIndexes_filtered = None
        
        self.Y_true = None
        self.Y_noisy = None
        self.Y_filtered = None
        self.W = None
        self.F = None
        self.out_dict = {}

    def run(self,hook_list=PLOT_HOOKS):
        for k,v in self.args.items():
            LOG.info("{}:{}".format(k,v),LOG.ll.EXPERIMENT)
        
        
        #Multiplex the arguments, allocating each to the correct step
        mplex = postprocess(keys_multiplex(self.args))
        
        
        #Get Hooks:
        hooks = select_and_add_hook(hook_list, mplex, self)
        
        
        
        
        LOG.info("Step 1: Read Dataset",LOG.ll.EXPERIMENT)
        
        #Select Input 
        self.X,self.W,  self.Y_true, self.labeledIndexes = select_input(**mplex["INPUT"])
        
        if self.W is None:
            self.W = select_affmat(**mplex["AFFMAT"]).generateAffMat(self.X,hook=hooks["AFFMAT"])
        
        
        
        if "know_estimated_freq" in mplex["ALG"].keys():
            mplex["ALG"]["useEstimatedFreq"] = np.sum(self.Y_true,axis=0) / self.Y_true.shape[0]
            mplex["ALG"].pop("know_estimated_freq")
        
        if "know_estimated_freq" in mplex["FILTER"].keys():
            mplex["FILTER"]["useEstimatedFreq"] = np.sum(self.Y_true,axis=0) / self.Y_true.shape[0]
            mplex["FILTER"].pop("know_estimated_freq")
            
        
        
        
        
        LOG.info("Step 2: Apply Noise",LOG.ll.EXPERIMENT)
        #Apply Noise
        self.Y_noisy = select_noise(**mplex["NOISE"]).corrupt(self.Y_true, self.labeledIndexes,hook=hooks["NOISE"])
        



        
        LOG.info("Step 3: Create Affinity Matrix",LOG.ll.EXPERIMENT)
        #Generate Affinity Matrix
        self.W = select_affmat(**mplex["AFFMAT"]).generateAffMat(self.X,hook=hooks["AFFMAT"])
        
        
        
        LOG.info("Step 4: Filtering",LOG.ll.EXPERIMENT)
        #Create Filter
        ft = select_filter(**mplex["FILTER"])
        self.ft = ft

        
        noisyIndexes = (np.argmax(self.Y_true,axis=1) != np.argmax(self.Y_noisy,axis=1))
        
        self.Y_filtered, self.labeledIndexes_filtered = ft.fit(self.X, self.Y_noisy, self.labeledIndexes, self.W, hook=hooks["FILTER"])
        
        
        LOG.info("Step 5: Classification",LOG.ll.EXPERIMENT)
        #Select Classifier 
        alg = select_classifier(**mplex["ALG"])
        #Get Classification
        self.F = alg.fit(self.X,self.W,self.Y_filtered,self.labeledIndexes_filtered,hook=hooks["ALG"])
        
        
        LOG.info("Step 6: Evaluation",LOG.ll.EXPERIMENT)
        LOG.debug("ALGORITHM settings:{}".format(mplex["ALG"]["algorithm"]),LOG.ll.EXPERIMENT)
        
        """ Accuracy. """
        acc = gutils.accuracy(gutils.get_pred(self.F), gutils.get_pred(self.Y_true))
        
        
        acc_unlabeled = gutils.accuracy(gutils.get_pred(self.F)[np.logical_not(self.labeledIndexes)],\
                                         gutils.get_pred(self.Y_true)[np.logical_not(self.labeledIndexes)])
        acc_labeled = gutils.accuracy(gutils.get_pred(self.F)[self.labeledIndexes],\
                                         gutils.get_pred(self.Y_true)[self.labeledIndexes])
        
        
        CMN_acc = gutils.accuracy(gutils.get_pred(gutils.class_mass_normalization(self.F,self.Y_filtered,self.labeledIndexes,normalize_rows=True)), gutils.get_pred(self.Y_true))
      
        
        """
            Log accuracy results and update output dictionary
        """
        def _log(msg):
            LOG.info(msg,LOG.ll.EXPERIMENT)
            
        _log("Accuracy: {:.3%} | {:.3%}".format(acc,1-acc))
        _log("Accuracy (unlabeled): {:.3%} |{:.3%}".format(acc_unlabeled,1-acc_unlabeled))
        _log("Accuracy (labeled): {:.3%} | {:.3%}".format(acc_labeled,1-acc_labeled))    
        _log("Accuracy w/ CMN: {:.3%} | {:.3%}".format(CMN_acc,1-CMN_acc))
        
        self.out_dict.update({OUTPUT_PREFIX + "acc" :acc})
        self.out_dict.update({OUTPUT_PREFIX + "acc_unlabeled" :acc_unlabeled})
        self.out_dict.update({OUTPUT_PREFIX + "acc_labeled" :acc_labeled})
        self.out_dict.update({OUTPUT_PREFIX + "CMN_acc" :CMN_acc})
        
        
        
        return self.out_dict
    
    

def run_debug_example_one(hook_list=[]):
    import tf_labelprop.experiment.specification.exp_chapelle as exp
    opt = exp.ExpChapelle("digit1").get_all_configs()[0]
    ExperimentRun(opt).run(hook_list=hook_list)
    
    
def run_debug_example_all():
    import tf_labelprop.experiment.specification.exp_chapelle as exp
    exp.ExpChapelle("").run_all()
    
def intcomp_demo():

    LOG.info("Demonstração para Inteligência Computacional")
    ds, alg = "", ""
    
    while not ds in ['mnist','isolet','g241c']: 
        LOG.info("Qual DATASET? (MNIST ou ISOLET)")
        ds = input().lower()
    while not alg in ['L','D','N']: 
        LOG.info("Qual LGCLVO? (L ou D ou N[enhum])")
        alg = input().upper()
    
    import tf_labelprop.experiment.specification.specification_bits as spec
    from tf_labelprop.experiment.specification.specification_bits import allPermutations as P
    
    from tf_labelprop.experiment.specification.specification_skeleton import EmptySpecification
    class ExpIntComp(EmptySpecification):
        CACHE_AFFMAT = False
        def __init__(self,ds,alg):
            self.ds = ds
            self.alg = alg

        def get_spec_name(self):
            return "INTCOMP"
        
        def generalConfig(self):
            s = spec.GENERAL_DEFAULT
            s['id'] = [1]
            return P(s)
        
        def inputConfig(self):
            if self.ds == 'mnist':
                s = spec.INPUT_MNIST
                s['labeled_percent'] = [100/70000]
            elif self.ds=="isolet":
                s = spec.INPUT_ISOLET
            else:
                s = spec.INPUT_CHAPELLE_A
                s["use_chapelle_splits"] = [True]
                s['num_labeled'] = [10]
                s['benchmark'] = ['g241c']
            return P(s)
    
        def filterConfig(self):
            def alpha_to_mu(alpha):
                return (1-alpha)/alpha
            if self.alg in ["L"]:
                s = spec.FILTER_LGC_LVO_AUTO
                s['LGC_iter'] = [1000]
            else:
                s = spec.FILTER_NOFILTER
            
            return  P(s)
        
        
        def noiseConfig(self):
            s = spec.NOISE_UNIFORM_DET_SOME
            s["corruption_level"] = [0.0 if self.ds == "mnist" else 0.0]
            return P(s)
        
        def affmatConfig(self):
            if self.ds == "g241c":
                s = spec.AFFMAT_DEFAULT
                s['k'] = [500]
            elif self.ds=="mnist":
                s = spec.AFFMAT_DEFAULT
                s['k'] = [15]
            elif self.ds == "isolet":
                s = spec.AFFMAT_ISOLET
            else:
                s = spec.AFFMAT_DEFAULT
            
            return P(s)
        def algConfig(self):
            if self.alg=="D":
                s =spec.ALGORITHM_LGCLOO_DEFAULT
                s["optimize_labels"] = [False]
                s["custom_conv"] = [False]
                
                s['p'] = [1500 if self.ds == 'isolet' else 50]
                return P(s)
            s = spec.ALGORITHM_GTAM_DEFAULT
            s['mu'] = [(1-0.99)/0.99]
            
            def alpha_to_mu(alpha):
                return (1-alpha)/alpha
            s["num_iter"] = [1000]
            return P(s)
    

    
    
    

    opt = ExpIntComp(ds=ds,alg=alg).get_all_configs()[0]
    
    ExperimentRun(opt).run(hook_list=[])
    
    
    
    
    
    
def main():
    
    import tf_labelprop.experiment.specification.specification_bits as spec
    from tf_labelprop.experiment.specification.specification_bits import allPermutations as P
    from tf_labelprop.experiment.specification.exp_chapelle import ExpChapelle
    class ExpChapelleLGCLVO(ExpChapelle):
        def get_spec_name(self):
            return "Experiment_Chapelle_{}_labels={}_split={}_alg=lgclvo".format(self.ds,self.num_labeled,self.use_chapelle_splits)
        def algConfig(self):
            s = spec.ALGORITHM_LGCLOO_DEFAULT
            s['optimize_labels'] = [False]
            s['custom_conv'] [True]
            s['p'] = [300]
            return P(s) 
        def noiseConfig(self):
            s = spec.NOISE_UNIFORM_DET_SOME
            return P(s)
    for ds in ["BCI","COIL","COIL2","digit1","g241c","g241n","USPS","Text"]:
        ExpChapelleLGCLVO(ds=ds,use_chapelle_splits=True).run_all()
    
    
def teste_():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TKagg")
    X = plt.imread('/home/klaus/Documents/parana_logo_2.png')
    _green = np.max(X[:,:,1])
    print("Max green: {}".format(_green))
    def get_class(x):
        if x[3] < 1:
            return -1
        elif x[2] >= 0.75*_green:
            return 1
        else:
            return 0
    
    X = np.apply_along_axis(lambda x: get_class(x),arr=X,axis = -1)
    Y = np.reshape(np.array(X),(-1,))
    X = np.array([[j ,X.shape[0] -1-i] for i in range(X.shape[0]) for j in range(X.shape[1])])
    
    X = X[Y>=0,:]
    Y = Y[Y>=0]
    np.random.seed(45)
    
    X = X + 2*np.random.random(size=X.shape)
    SAMPLE_NUM = 750
    perm = np.random.permutation(np.arange(X.shape[0]))[:SAMPLE_NUM]
    perm = np.arange(X.shape[0])[::int(X.shape[0]/SAMPLE_NUM)]
    X = X[perm,:]
    Y = Y[perm]
    
    print(X.shape)
    X  = X.astype(np.float32)
    W = AffMatGenerator(dist_func='constant',mask_func='mutknn',k=13).generateAffMat(X, None)
    
    print(W.__class__)
    import tf_labelprop.output.plots as pcore
    pcore.plot_all_indexes(X, Y, labeledIndexes=[True]*X.shape[0], W=W,plot_filepath= '/home/klaus/Documents/parana_plot_2.png')
    
    plt.show()
    
if __name__ == "__main__":
    run_debug_example_one()
    #intcomp_demo()
    #teste()
    

    
    
    
