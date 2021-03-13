'''
Created on 27 de mar de 2019

@author: klaus
'''
import numpy as np
from tf_labelprop.experiment.selector import select_input
from tf_labelprop.experiment.specification.specification_bits import allPermutations as P, \
    AFFMAT_DEFAULT
import tf_labelprop.experiment.specification.specification_bits as spec
from tf_labelprop.experiment.specification.specification_skeleton import EmptySpecification


class ExpChapelle(EmptySpecification):
    '''
    classdocs
    '''
    ds = 'all'
    
    def get_spec_name(self):
        return "Experiment_Chapelle_{}_labels={}_split={}".format(self.ds,self.num_labeled,self.use_chapelle_splits)
    
    def generalConfig(self):
        s = spec.GENERAL_DEFAULT
        if "SecStr" in self.ds or "SecStrExtra" in self.ds:
            s['id'] = list(range(10))
        else:
            s['id'] = list(range(12) )
        return P(s)
    
    def inputConfig(self):
        s = spec.INPUT_CHAPELLE_A
        s["benchmark"] = self.ds
        s["use_chapelle_splits"] = [self.use_chapelle_splits]
        s['num_labeled'] = [self.num_labeled]
        return P(s)

    def filterConfig(self):
        s = spec.FILTER_NOFILTER
        return  P(s)
    
    
    def noiseConfig(self):
        s = spec.NOISE_UNIFORM_DET_NONE
        return P(s)
    
    def affmatConfig(self):
        s = spec.AFFMAT_DEFAULT
        
        return P(s) + P(spec.AFFMAT_CONSTANT)
    def algConfig(self):
        s = spec.ALGORITHM_NONE
        return P(s)

    def __init__(self,ds,use_chapelle_splits=True,num_labeled=100):
        self.use_chapelle_splits = use_chapelle_splits
        self.num_labeled = num_labeled
        if ds == 'all':
            ds = ["BCI","COIL","COIL2","digit1","g241c","g241n","USPS","Text"]
        elif ds == 'reduced':
            ds = ["BCI","COIL","COIL2","digit1","g241c","g241n","USPS"]
        else:
            if not isinstance(ds,list):
                ds = [ds]
            assert isinstance(ds[0],str)
        self.ds = ds

    