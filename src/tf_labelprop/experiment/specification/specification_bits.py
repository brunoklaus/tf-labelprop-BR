'''
Created on 28 de mar de 2019

@author: klaus
'''

'''
::

Utility Functions
----------------------

'''

from functools import reduce
from os import path

import numpy as np
import tf_labelprop
from tf_labelprop.gssl.classifiers import LGC_LVO_AUTO_D
import tf_labelprop.input.dataset as ds


def __prod(a,b):
    DICTS = []
    newvals = b[1]
    newkeys = b[0]
    for d in a:
        for i in np.arange(len(newvals)): 
            K = list(d.keys()) + [newkeys]
            V = list(d.values()) + [newvals[i]] 
            newDict = {}
            for k,v in zip(K,V):
                newDict[k] = v
            DICTS += [newDict]
    return(DICTS)

def allPermutations(args):
    """ Given a dictionary with lists as values, produce every dictionary possible when picking exactly one value
    from each list. 
    
    For example, if there are two keys, each linked to a list of 4 values, the returned list will have size 4*4=16.
    
    Args:
        args (dict) : A dictionary, such that every value is a list.
    Returns:
        List[dict] : A list comprised of every permutation when picking exactly one value from each list of the original dictionary. 
    """
    return(list(reduce(lambda a,b: __prod(a,b), list(zip(args.keys(),args.values())), [{}])))

def comb(dict_A,dict_B):
    """ Combine two lists of dictionaries.
    
        Args: 
            dict_A (List[Dict]): First list.
            dict_B (List[Dict]): Second list.
            
        Returns:
            List[Dict]:  all possible dictionaries when extending a dict from the 1st list with one from the 2nd.
    """
    if not isinstance(dict_A,list):
        raise "ERROR: dict_A, dict_B must each be a list of dicts"
    if not isinstance(dict_B,list):
        raise "ERROR: dict_A, dict_B must each be a list of dicts"
    
        
    l = [None] * (len(dict_A) * len(dict_B))
    i = 0
    for a in dict_A:
        for b in dict_B:
            comb_dict = {}
            for k,v in zip(list(a.keys()) + list(b.keys()), list(a.values()) + list(b.values())):
                comb_dict[k] = v
            l[i] = comb_dict
            i += 1
    assert i == (len(dict_A) * len(dict_B))
    return l

def add_key_prefix(pref,dictionary):
    """ Adds a prefix to each key in a dictionary."""
    x = dict()
    for k,v in dictionary.items():
        x[pref+k] = v
    return(x)   


"""
Input configs
---------------------------------------
"""
INPUT_GAUSSIANS_DYNAMIC = {
        "dataset": ["sk_gaussian"],
        "dataset_sd": [0.4,1,2,3],
        "labeled_percent": [0.1,0.05,0.025],
        }

INPUT_SPIRALS_DYNAMIC = {
        "dataset": ["sk_spiral"],
        "dataset_sd": [0.04,0.15,0.2,0.3],
        "labeled_percent": [0.1],
        }

INPUT_SPIRALS_FIXED = {
        "dataset": ["spiral"],
        "labeled_percent": [0.1],
        }
INPUT_MNIST = {
        "dataset": [ds.MNIST],
        "labeled_percent": [100/70000],
        }


INPUT_CHAPELLE_A = {
        "dataset": [ds.ChapelleDataset],
        "benchmark":["digit1","COIL","USPS","BCI","g241c","g241d"],
        "num_labeled": [10,100],
        "use_chapelle_splits":[True]
        }

INPUT_ISOLET = {
        "dataset": [ds.ISOLET],
        "labeled_percent": [1040/7797],
        }


INPUT_CIFAR_10 = {
        "dataset": ["cifar10"],
        "labeled_percent": [0.1,0.05,0.025,0.01],
        }


"""
Affinity matrix configs
----------------------------------
"""
AFFMAT_DEFAULT = {
        "k": [15],
        "mask_func": ["knn"],
        "sigma": ["mean"],
        "dist_func":["gaussian"],
        "row_normalize":[False],
        }
AFFMAT_ISOLET = {
        "k": [10],
        "mask_func": ["knn"],
        "sigma": [100.0],
        "dist_func":["gaussian"]
        }
AFFMAT_CONSTANT = {
        "k": [15],
        "mask_func": ["knn"],
        "dist_func":["constant"],
        "row_normalize": [False],
        }



"""
Noise configs
-------------------------------
"""
NOISE_UNIFORM_DET_MANY = {
        "corruption_level" : [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4],
        "type" : ["NCAR"],
        "deterministic" : [False]
        }

NOISE_UNIFORM_DET_SOME = {
        "corruption_level" : [0,0.1,0.2,0.35],
        "type" : ["NCAR"],
        "deterministic" : [False]
        }
NOISE_UNIFORM_DET_NONE = {
        "corruption_level" : [0],
        "type" : ["NCAR"],
        "deterministic" : [False]
        }


NOISE_UNIFORM_DET_MODERATE = {
        "corruption_level" : [0.2],
        "type" : ["NCAR"],
        "deterministic" : [False]
        }



"""
Filter configs
------------------------------
"""
FILTER_NOFILTER = {
        "filter": [None], 
        }



FILTER_LDST = {
        "filter": ["LDST"],
        "mu": [0.1111,99.0,9.0,0.1111],
        "useZ": [True],
         "tuning_iter": [1.00],
         "tuning_iter_as_pct":[True],
         "constantProp":[False],
         "gradient_fix":[True],
         "weigh_by_degree":[True,False]
        }

FILTER_LGC_LVO = {
        "filter": ["LGC_LVO"],
        "mu": [0.1111,99.0,9.0,0.1111],
         "tuning_iter": [1000],
         "tuning_iter_as_pct":[False],
         "constantProp":[False],
         "useZ":[False],
        "normalize_rows":[True,False]
        }



FILTER_LGC_LVO_AUTO = {
        "filter": ["LGC_LVO_AUTO"],
        "mu": [0.1111,99.0,9.0,0.1111],
        "loss": ["xent","mse"]
        }

FILTER_LDST_CONSTPROP = {
        "filter": ["LDST"],
        "mu": [0.0101,99.0,9.0,0.1111],
         "tuning_iter": [1.00,0.75,0.5,0.25,0],
         "tuning_iter_as_pct":[True],
         "constantProp":[True]
        }

FILTER_MR = {
        "filter": ["MRF"],
        "p": [1,4,15],
         "tuning_iter": [1.00,0.75,0.5,0.25,0],
         "tuning_iter_as_pct":[True],
        }


"""
Algorithm (that is, Classifier) configs
------------------------------------------
"""
ALGORITHM_GTAM_DEFAULT = {
   "algorithm" : ["GTAM"],
   "weigh_by_degree":[True,False],
   "mu":[0.001,0.0101,0.1111,9,99.0]  
}

ALGORITHM_LGCLOO_DEFAULT = {
   "algorithm" : [LGC_LVO_AUTO_D],
   "optimize_labels" : [False,True],
   "custom_conv": [False,True],
   "p":[100]
}

ALGORITHM_GFHF_DEFAULT = {
   "algorithm" : ["GFHF"]
}



ALGORITHM_LGC_DEFAULT = {
   "algorithm" : ["LGC"],
   "alpha" : [0.9,0.99,0.1,0.999]  
}

ALGORITHM_CLGC_DEFAULT = {
   "algorithm" : ["CLGC"],
   "use_estimated_freq" : [True,False],
   "alpha" : [0.9,0.99,0.1,0.999]  
}


ALGORITHM_SIIS_DEFAULT = {
   "m" : [30],
   "algorithm" : ["SIIS"],
   "alpha" : [100],
   "beta" : [10]  
}

ALGORITHM_RF_DEFAULT = {
   "algorithm" : ["RF"],
   "n_estimators" : [100]  
}



ALGORITHM_GTAM_DEFAULT = {
   "algorithm" : ["GTAM"],
   "mu" : [0.1111,0.001,9,99,1],
   "constantProp":[False],   
    "know_estimated_freq":[False],
    "use_estimated_freq":[True,False],
}




ALGORITHM_NONE = {
   "algorithm" : [None]
}



"""
General configs
-------------------------------------
"""
GENERAL_DEFAULT = {
        "plot_mode": ["ignore"],
        "id": np.arange(20),
        }

