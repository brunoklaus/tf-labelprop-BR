'''
Created on 28 de mar de 2019

@author: klaus
'''
from enum import Enum  
from inspect import signature 
from math import sqrt
import numpy as np
import os.path as path
from sklearn.datasets import make_blobs, make_moons
from tf_labelprop.experiment.hooks import hook_skeleton as hks
from tf_labelprop.experiment.hooks import ldst_filterstats_hook
from tf_labelprop.experiment.hooks import plot_hooks as plt_hks
from tf_labelprop.experiment.hooks import time_hook as t_hks
from tf_labelprop.experiment.prefixes import OUTPUT_PREFIX
from tf_labelprop.gssl.classifiers import GSSLClassifier
from tf_labelprop.gssl.classifiers import LGC_LVO_AUTO_D
from tf_labelprop.gssl.classifiers import LGC_LVO_AUTO_D
from tf_labelprop.gssl.graph.gssl_affmat import AffMatGenerator
import tf_labelprop.gssl.graph.gssl_utils as gutils
from tf_labelprop.input.dataset import ChapelleDataset
from tf_labelprop.input.dataset import GSSLDataset
import tf_labelprop.input.dataset._toy_ds as toyds
from tf_labelprop.input.dataset.cifar10 import get_cifar10
from tf_labelprop.input.noise.noise_process import LabelNoiseProcess
from tf_labelprop.output.folders import PLOT_FOLDER


def select_input(dataset,seed,use_chapelle_splits=False,labeled_percent=None,num_labeled=None,**kwargs):
    """ Gets the input dataset, according to some specification.
    
    Currently, the following keyword arguments are required:
    
        * dataset : identifies the dataset. Currently, this may be
        
            1. The name of any of the toy datasets.
            2. `sk_gaussian` to use `sklearn's` ``make_blob`` command at runtime.
               requires ``dataset_sd`` config to determine the dispersion.            
            3. `sk_spiral` to use `sklearn's` ``make_moons`` command at runtime.
               requires ``dataset_sd`` config to determine the dispersion.
        
        * seed : Specifies the seed for reproducibility purposes.
        * labeled_percent or num_labeled : Specifies the percentage/amount of instances to be marked as 'labeled'.
        
        Args:
            `**kwargs`: Key-value pairs with the configuration options of the input.
            
        Returns:
            (tuple): tuple containing:
                1. (`NDArray[float].shape[N,D]`) : An input matrix, describing N instances of dimension D.
                2. (Union[:class:`tf_labelprop.gssl.graph.gssl_affmat.AffMat`,None]) : An affinity matrix, when applicable.
                3. (`NDArray[float].shape[N,C]`) : A belief matrix corresponding to the clean labels. Every row is one-hot, marking down the correct label.
                4. (`NDArray[bool].shape[N]`): A boolean array, indicating which instances are to be interpreted as labeled.
        
        Raises:
            KeyError: If one of the required keys is not found.
    """
    
    """
    -------------------------------------------------------------------
        Read Dataset X,Y
    -------------------------------------------------------------------
    """
    assert issubclass(dataset,GSSLDataset)
    kwargs = dict(kwargs)
    
    if dataset == ChapelleDataset and use_chapelle_splits:

        if (num_labeled is None):
            raise KeyError("To use Chapelle's datasets with the custom benchmark splits, please specify `num_labeled` directly")
        ds = dataset(split=seed,labels=num_labeled,use_splits=True,**kwargs)
        ds_x, _, ds_y = ds.load()
        labeledIndexes = np.array(ds.labeledIndexes)

        
    else:
        ds_x, _, ds_y = dataset(**kwargs).load()
        if len(ds_y.shape) == 1:
            print(ds_y.shape)
            print(np.min(ds_y))
            
            print(np.max(ds_y))
            
            ds_y = gutils.init_matrix(ds_y, ds_y >= 0)
        """
        -------------------------------------------------------------------
            Define Labeled Indices
        -------------------------------------------------------------------
        """
        _where_known = np.where(np.max(ds_y,axis=1) > 0 )[0]
        
        if not num_labeled is None:
            labeled_percent = num_labeled / len(_where_known) 
        if (num_labeled is None) and (labeled_percent is None):
            raise KeyError("Please use 'labeled_percent' or 'num_labeled' as a key in the input specification.")
        
        print(labeled_percent)
        
        labeledIndexes = np.array([False]*ds_y.shape[0])
        

        
        labeledIndexes[_where_known] = gutils.split_indices(ds_y[_where_known,:], labeled_percent,seed)
        print(np.mean(labeledIndexes))
    
    return ds_x.astype(np.float32), _, ds_y.astype(np.float32), labeledIndexes

def select_noise(**kwargs):
    """ Delegates to the appropriate  noise process.
 
        Currently, we simply forward it to :py:class:`input.noise.noise_process.LabelNoiseProcess`.
        More extensive documentation on the label noise process specification can be found there, as well.
        
        Args:
        `**kwargs`: configuration of affinity matrix
        
        Returns:
            (`LabelNoiseProcess`) : The appropriately configured label noise process.
         
    """
    return LabelNoiseProcess(**kwargs)



def select_affmat(**kwargs):
    """ Delegates to the appropriate Affinity Matrix generator.
 
        Currently, we simply forward it to :py:class:`gssl.graph.gssl_affmat.AffMatGenerator`.
        More extensive documentation on the affinity matrix specification can be found there, as well.
        
        Args:
        `**kwargs`: configuration of affinity matrix
        
        Returns:
            (`AffMatGenerator`) : The appropriately configured affinity matrix generator
         
    """
    return AffMatGenerator(**kwargs)

def select_classifier(**kwargs):
    """ Delegates to the appropriate classifier.
    
    Currently available classifiers:
    
        * LGC (Local and Global Consistency) : :class:`gssl.classifiers.LGC.LGCClassifier`
        * GTAMClassifier (Graph Transduction via Alternating Minimization) : :class:`gssl.classifiers.GTAMClassifier.GTAMClassifier`
        * GFHF (Gaussian Fields and Harmonic Functions ) : :class:`gssl.classifiers.GFHF.GFHFlassifier`
    
    Args:
        `**kwargs`: configuration of classifier
    Returns:
        (`GSSLClassifier`) : The appropriately configured classifier
    
    """
    if not "algorithm" in kwargs.keys():
        raise KeyError("Key {} not found".format("algorithm"))
    alg = kwargs.pop("algorithm")
    
    import tf_labelprop.gssl.classifiers as cl
    
    
    if alg is None:
        return cl.GSSLClassifier(**kwargs)
    elif isinstance(alg,str):
        alg = alg.upper()
        switcher = {
                "LGC": lambda : cl.LGCClassifier(**kwargs),
                "CLGC": lambda :cl.CLGC_Classifier(**kwargs),
                "GTAM":lambda: cl.GTAMClassifier(**kwargs),            
                "GFHF": lambda: cl.GFHF(**kwargs),  
                "LGCLOO": lambda: cl.LGC_LVO_AUTO_D(**kwargs),
                "SIIS": lambda: cl.SIISClassifier(**kwargs),
                "RF": lambda: cl.RandomForest(**kwargs),
                "NONE":lambda:cl.GSSLClassifier(**kwargs)         
                }
        if not alg in switcher.keys():
            raise KeyError("Did not find {} as a selector option".format(kwargs["algorithm"]))
        return switcher[alg]()

    else:
        if not issubclass(alg,cl.GSSLClassifier):
            raise ValueError("Invalid type: {}".format(type(alg)))
        return alg(**kwargs)
    
def select_filter(**kwargs):
    """ Delegates to the appropriate filter.
    
    Currently available classifiers:
    
        * LDST (Label Diagnosis through self tuning):  :class:`gssl.filters.LDST.LDST`
            
    Args:
        `**kwargs`: configuration of classifier
    Returns:
        (`GSSLFilter`) : The appropriately configured filter
    
    """
    if not "filter" in kwargs.keys():
        raise KeyError("Key {} not found".format("filter"))
    ft = kwargs.pop("filter")
    import tf_labelprop.gssl.filters as fl
    if ft is None:
        return fl.GSSLFilter(**kwargs)
    elif isinstance(ft,str):
        ft = ft.upper()
        switcher = {
                "MRF": lambda : fl.MRRemover(**kwargs),
                "LDST":lambda : fl.LDSTRemover(**kwargs),
                "LGC_LVO": lambda: fl.LGC_LVO_Filter(**kwargs),
                "LGC_LVO_AUTO": lambda: fl.LGC_LVO_AUTO_Filter(**kwargs),
                "NONE":lambda:fl.GSSLFilter(**kwargs),
                        
        }
        if not ft in switcher.keys():
            raise KeyError("Did not find {} as a selector option".format(kwargs["filter"]))
        
        return switcher[ft]()
    else:
        if not issubclass(ft,fl.GSSLFilter):
            raise ValueError("Invalid type: {}".format(type(ft)))
        return ft(**kwargs)

class HookTimes(Enum):
    """
        Information used to infer at which point of the execution a hook takes place.
    """
    BEFORE_NOISE = 0
    DURING_NOISE = 1
    AFTER_NOISE = 2
    BEFORE_AFFMAT = 3
    DURING_AFFMAT = 4
    AFTER_AFFMAT = 5
    BEFORE_FILTER = 6
    DURING_FILTER = 7
    AFTER_FILTER = 8
    BEFORE_CLASSIFIER = 9
    DURING_CLASSIFIER = 10
    AFTER_CLASSIFIER = 11

__hook_plotnames = {'INIT_ALL':'clean_all.png',
                'INIT_LABELED':'clean_labeled.png',
                'NOISE_AFTER':'after_noise.png',
                
                'W_INIT_ALL':'clean_all.png',
                'W_INIT_LABELED':'clean_labeled.png',
                'W_NOISE_AFTER':'noisy_labeled.png',
                "W_FILTER_AFTER":'after_filter.png',
                "FILTER_ITER":'filter_iter.png',
                
                
                "FILTER_AFTER":'after_filter.png',
                "ALG_RESULT":'after_classifier.png',
                "ALG_ITER":'alg_iter.mp4',
                "GTAM_Y":'gtam_y_iter.mp4',
                "GTAM_F":'gtam_f_iter.mp4',
                "GTAM_Q":'gtam_q_iter.mp4',
                }

Hook = Enum('Hook',' '.join(list(__hook_plotnames.keys()) + ["T_NOISE","T_FILTER","T_ALG","T_AFFMAT","LDST_STATS_HOOK"] ) )
__hook_times = {'INIT_ALL':HookTimes.BEFORE_NOISE,
                'INIT_LABELED':HookTimes.BEFORE_NOISE,
                'NOISE_AFTER':HookTimes.AFTER_NOISE,
                "FILTER_AFTER":HookTimes.AFTER_FILTER,
                
                'W_INIT_ALL':HookTimes.BEFORE_FILTER,
                'W_INIT_LABELED':HookTimes.BEFORE_FILTER,
                'W_NOISE_AFTER':HookTimes.BEFORE_FILTER,
                "W_FILTER_AFTER":HookTimes.AFTER_FILTER,
                "FILTER_ITER":HookTimes.DURING_FILTER,
                
                
                "ALG_RESULT":
                HookTimes.AFTER_CLASSIFIER,
                "ALG_ITER":HookTimes.DURING_CLASSIFIER,
                "GTAM_Y": HookTimes.DURING_CLASSIFIER,
                "GTAM_F":HookTimes.DURING_CLASSIFIER,
                "GTAM_Q":HookTimes.DURING_CLASSIFIER,
                }
def _get_hook_plot_fname(hook,spec_name):
    return path.join(PLOT_FOLDER,spec_name,"t"+str(__hook_times[hook.name].value)+"_"+__hook_plotnames[hook.name])

    
    
def select_and_add_hook(hook_list, mplex, experiment=None):
    """ Obtains the relevant hook for the classifier/filter.
    
    Each hook is associated with a filename prefix, which is partially determined by the prefix t#, where # is given according to the enum :class:`experiment.selector.HookTimes`
    Currently available hooks:
    
        * **Hook.INIT_ALL**: At HookTimes.BEFORE_NOISE, plots true labels of all instances
        * **Hook.INIT_LABELED**: At HookTimes.BEFORE_NOISE, plots true labels of instances marked as 'labeled'
        * **Hook.NOISE_AFTER**: At HookTimes.AFTER_NOISE, plots corrupted labels
        * **Hook.W_INIT_ALL** : At HookTimes.BEFORE_FILTER, plots true labels of all instances, with the affinity matrix.
        * **Hook.W_INIT_LABELED** : At HookTimes.BEFORE_FILTER, plots true labels of instances marked as 'labeled', with the affinity matrix.
        * **Hook.W_NOISE_AFTER** : At HookTimes.BEFORE_FILTER, plots corrupted labels, with the affinity matrix.
        * **Hook.FILTER_AFTER**: At HookTimes.AFTER_FILTER, plots filtered labels
        * **Hook.FILTER_ITER**: At HookTimes.DURING_FILTER, plots the filtered labels at each step.
        * **Hook.ALG_ITER**: At HookTimes.DURING_CLASSIFIER, plots the classification  at each step.
        * **Hook.ALG_RESULT**: At HookTimes.AFTER_CLASSIFIER, plots classification of algorithm.
        * **Hook.T_NOISE**: At HookTimes.AFTER_NOISE, adds time taken by the noise process to the output dictionary.        
        * **Hook.T_AFFMAT**: At HookTimes.AFTER_AFFMAT, adds time taken by the affinity matrix generation to the output dictionary.
        * **Hook.T_FILTER**: At HookTimes.AFTER_FILTER,  adds time taken by filtering process to the output dictionary.
        * **Hook.T_ALG**: At HookTimes.AFTER_CLASSIFIER, adds time taken by the classification to the output dictionary.
        
        
        
        
    
    Args:
        hook_mode (List[int]) : the identifier for the hook
        mplex (Dict[Dict]]) : The multiplexed configs. See :meth:`experiment.experiments.keys_multiplex`
        experiment (Experiment) : The experiment calling this function. Could be used for callbacks for certain hooks.
    Returns:
        (`Dict[GSSLHook]`) : The appropriately configured hook   
    """
    if not isinstance(hook_list,list):
        raise ValueError("hook_list must be a list")
    

    l = dict()
    for k in mplex.keys():
        l[k] = list()
        
    for h in hook_list:
        if h.name in list(__hook_plotnames.keys()):
            fname = _get_hook_plot_fname(h,mplex["GENERAL"]["spec_name"])
            if h == Hook.INIT_LABELED:
                l["NOISE"].append(plt_hks.plotHook(filename_path=fname,title="Observed Labels:",plot_W=False,
                                                          when="begin",experiment=experiment, only_labeled=True)) 
            elif h == Hook.INIT_ALL:
                l["NOISE"].append(plt_hks.plotHook(filename_path=fname,title="Ground Truth:",plot_W=False,
                                                          when="begin",experiment=experiment,only_labeled=False)) 
            elif h == Hook.W_INIT_ALL:
                l["ALG"].append(plt_hks.plotHook(filename_path=fname,title="All clean labels + Affmat:",plot_W=True,
                                                          when="begin",experiment=experiment,only_labeled=False,
                                                          force_Y_callback="Y_true",force_lb_callback="labeledIndexes"))
            elif h == Hook.W_INIT_LABELED:
                l["ALG"].append(plt_hks.plotHook(filename_path=fname,title="Clean labeled instances + Affmat:",plot_W=True,
                                                          when="begin",experiment=experiment,only_labeled=True,
                                                          force_Y_callback="Y_true",force_lb_callback="labeledIndexes"))  
            elif h == Hook.NOISE_AFTER:
                l["NOISE"].append(plt_hks.plotHook(filename_path=fname,title="Noisy labels:",plot_W=False,
                                                          when="end",experiment=experiment,only_labeled=True)) 
            elif h == Hook.FILTER_AFTER:
                l["FILTER"].append(plt_hks.plotHook(filename_path=fname,title="Filtered labels:",plot_W=False,
                                                          when="end",experiment=experiment,only_labeled=True)) 
                
            elif h == Hook.W_NOISE_AFTER:
                l["ALG"].append(plt_hks.plotHook(filename_path=fname,title="Noisy labels + Affmat:",plot_W=True,
                                                          when="end",experiment=experiment,only_labeled=True,
                                                          force_Y_callback="Y_noisy",force_lb_callback="labeledIndexes"))
            elif h == Hook.W_FILTER_AFTER:
                l["FILTER"].append(plt_hks.plotHook(filename_path=fname,title="Filtered labels + Affmat:",plot_W=True,
                                                          when="end",experiment=experiment,only_labeled=True,
                                                          force_Y_callback=None)) 
            
            elif h in [Hook.GTAM_Y,Hook.GTAM_F,Hook.GTAM_Q]:
                if mplex["ALG"]["algorithm"] == "GTAMClassifier":
                    if h == Hook.GTAM_Y:
                        l["ALG"].append(plt_hks.plotIterGTAMHook(mode="Y",title="GTAMClassifier Y:",video_path=fname,experiment=experiment))
                    elif h == Hook.GTAM_F:
                        l["ALG"].append(plt_hks.plotIterGTAMHook(mode="F",title="GTAMClassifier F:",video_path=fname,experiment=experiment))
                    else:
                        l["ALG"].append(plt_hks.plotIterGTAMHook(mode="Q",title="GTAMClassifier Q:",palette="BuPu_r",video_path=fname,experiment=experiment))
                        
            elif h == Hook.FILTER_ITER:
                if mplex["FILTER"]["filter"] in ["LGC_LVO","LDST"]:
                    l["FILTER"].append(plt_hks.plotIterHook(video_path=fname,title="Filtered labels:",experiment=experiment,plot_W=False,
                                             only_labeled=True,plot_mode="discrete",step_size=1,create_video=False,
                                             keep_images=True,temp_subfolder_name="filter_iter")) 
            
            elif h == Hook.ALG_ITER:
                if mplex["ALG"]["algorithm"] in ["MREG"]:
                    l["ALG"].append(plt_hks.plotIterHook(video_path=fname,title="Eigenfunctions:",experiment=experiment,plot_W=False,
                                             only_labeled=False,plot_mode="continuous",step_size=1,create_video=False,
                                             keep_images=True)) 
                     
                elif mplex["ALG"]["algorithm"] == "GTAMClassifier":
                    pass
                elif mplex["ALG"]["algorithm"] == "LGC":
                    l["ALG"].append(plt_hks.plotIterHook(video_path=fname,title="Classification:",experiment=experiment,plot_W=False,
                                             only_labeled=True,plot_mode="discrete",create_video=True)) 
                else:
                    l["ALG"].append(plt_hks.plotIterHook(video_path=fname,title="Classification:",experiment=experiment,plot_W=False,
                                             only_labeled=False,plot_mode="discrete",create_video=True)) 
            elif h == Hook.ALG_RESULT:
                l["ALG"].append(plt_hks.plotHook(filename_path=fname,title="Classification:",experiment=experiment,plot_W=False,
                                         when="end",only_labeled=False))
            else:
                raise ValueError("Unknown hook")
        else:
            if h == Hook.T_NOISE:
                l["NOISE"].append(t_hks.timeHook(experiment, timer_name=OUTPUT_PREFIX + "noise_time"))
            elif h == Hook.T_FILTER:
                l["FILTER"].append(t_hks.timeHook(experiment, timer_name=OUTPUT_PREFIX + "filter_time"))
            elif h == Hook.T_ALG:
                l["ALG"].append(t_hks.timeHook(experiment, timer_name=OUTPUT_PREFIX + "classifier_time"))
            elif h == Hook.T_AFFMAT:
                l["AFFMAT"].append(t_hks.timeHook(experiment, timer_name=OUTPUT_PREFIX + "affmat_time"))
            elif h == Hook.LDST_STATS_HOOK:
                l["FILTER"].append(ldst_filterstats_hook.LDSTFilterStatsHook(experiment,step_size=1))
            else:
                raise ValueError("Unknown hook {}".format(h))
             
    hook_dict = {}
    for k in mplex.keys():
        temp = l[k]
        if len(temp) == 0:
            hook_dict[k] = None
        elif len(temp) == 1:
            hook_dict[k] = temp[0]
        else:
            hook_dict[k] = hks.CompositeHook(temp)
    
            
    return hook_dict