'''
Created on 27 de mar de 2019

@author: klaus
'''
from functools import reduce
from functools import wraps
from multiprocessing import Process, Queue
import os
import shutil
import sys 
import traceback

import progressbar

import numpy as np
import pandas as pd
from tf_labelprop.experiment.experiments import AFFMAT_PREFIX, GENERAL_PREFIX, ALG_PREFIX, NOISE_PREFIX, INPUT_PREFIX, FILTER_PREFIX, \
    ExperimentRun
from tf_labelprop.experiment.experiments import TIME_HOOKS
from tf_labelprop.experiment.prefixes import AFFMAT_PREFIX, NOISE_PREFIX, \
    GENERAL_PREFIX
import tf_labelprop.experiment.specification.specification_bits as spec
from tf_labelprop.gssl.classifiers import GSSLClassifier
from tf_labelprop.gssl.filters import GSSLFilter
from tf_labelprop.input.dataset import GSSLDataset
import tf_labelprop.logging.logger as LOG
from tf_labelprop.output.aggregate_csv import aggregate_csv
from tf_labelprop.output.folders import CSV_FOLDER


def processify(func):
        '''
        From https://gist.github.com/schlamar/2311116
        Decorator to run a function as a process.
        Be sure that every argument and the return value
        is *pickable*.
        The created process is joined, so the code does not
        run in parallel.
        '''
    
        def process_func(q, *args, **kwargs):
            try:
                ret = func(*args, **kwargs)
            except Exception:
                ex_type, ex_value, tb = sys.exc_info()
                error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
                ret = None
            else:
                error = None
    
            q.put((ret, error))
    
        # register original function with different name
        # in sys.modules so it is pickable
        process_func.__name__ = func.__name__ + 'processify_func'
        setattr(sys.modules[__name__], process_func.__name__, process_func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            q = Queue()
            p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
            p.start()
            ret, error = q.get()
            p.join()
    
            if error:
                ex_type, ex_value, tb_str = error
                message = '%s (in subprocess)\n%s' % (ex_value, tb_str)
                raise ex_type(message)
    
            return ret
        return wrapper
class EmptySpecification(object):
    """ EmptySpecification defines the methods expected from any class representing a 
    specification of experiments. By itself, it also specifies an empty set of experiments.
    """
    
    
    
    FORCE_GTAM_LDST_SAME_MU = True
    TUNING_ITER_AS_NOISE_PCT = False
    
    CACHE_AFFMAT = True
    CLEAN_AFFMAT_DIR = True
    LABELS_INFLUENCE_AFFMAT = False 


    WRITE_FREQ = 10000
    DEBUG_MODE = True
    OVERWRITE = True
    
    experimentrun_type = ExperimentRun
    
    def get_spec_name(self):
        """ Gets the name identifying the set of experiments that come out of this specification."""
        return ""
    
    def generalConfig(self):
        """General configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    def inputConfig(self):
        """Input configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    def noiseConfig(self):
        """Noise process configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    def filterConfig(self):
        """Filter configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    
    
    
    def affmatConfig(self):
        """Input configuration.
        
        Returns:
            `List[dict]` A list of dictionaries, one for each possible configuration.
        """
        return [{}]
    

    def algConfig(self):
        """Input configuration.
        
        Returns:
            `dict` A dictionary, such that each key maps to a list containing each possible value
            that the attribute might take.
        """
        return [{}]
    
    
    
    def remove_undesirable_configs(self,res):
        """Removes undesirable configurations or performs other postprocessing adjustments to the list of configurations.
        For example, if we want to have the LDST filter and LGC algorithms to have the same parameter MU, we can remove configs where they do not match. Moreover, if we want
        to have the number of tuning iterations of LDST to be a fraction of the amount of noise, we can manually instanstiate this number over each list element.
        
            Args:
                res (List[Dict]): A list of all possible configs
            Returns:
            `List[dict]` An updated list of all possible configs. 
        """
        
        if self.FORCE_GTAM_LDST_SAME_MU:
            old_len = len(res)
            res = [x for x in res if \
                   not (x[ALG_PREFIX+"algorithm"]=="GTAM" and \
                   x[FILTER_PREFIX+"filter"]=="LDST" and x[ALG_PREFIX+"mu"] != x[FILTER_PREFIX+"mu"])]
            
            res = [x for x in res if \
                   not (x[ALG_PREFIX+"algorithm"]=="LGC" and \
                   x[FILTER_PREFIX+"filter"]in["LDST","LGC_LVO"] and np.round((1-x[ALG_PREFIX+"alpha"])/x[ALG_PREFIX+"alpha"],4) != np.round(x[FILTER_PREFIX+"mu"],4))]
            
            LOG.debug("Number of configurations removed due to forcing GTAM have same mu param as filter: {}"\
                      .format(old_len-len(res)),LOG.ll.SPECIFICATION)
            
        if self.TUNING_ITER_AS_NOISE_PCT:
            for x in res:
                if not FILTER_PREFIX+ "tuning_iter" in x.keys():
                    pass
                else:
                    x[FILTER_PREFIX + "tuning_iter" ] = x[INPUT_PREFIX+"labeled_percent"] *\
                                         x[NOISE_PREFIX+"corruption_level"] *\
                                         x[FILTER_PREFIX+"tuning_iter"]
                    x[FILTER_PREFIX + "tuning_iter_as_pct"] = True
        return res
    
    def get_keys_relevant_to_affmat(self):
        def with_prefix(l,pref):
            return [pref+x for x in l]
        
        def get_keys(l):
            return list(pd.unique(np.concatenate([list(x.keys()) for x in l]) ))
            
        relevant_keys_to_affmat = with_prefix(get_keys(self.affmatConfig()),pref=AFFMAT_PREFIX)

        inp_l =  get_keys(self.inputConfig())

        if not self.LABELS_INFLUENCE_AFFMAT:
            [inp_l.remove(x) for x in ['labeled_percent','num_labeled'] if x in inp_l]


        relevant_keys_to_affmat.extend(with_prefix(inp_l,pref=INPUT_PREFIX)) 
        
        if  self.LABELS_INFLUENCE_AFFMAT:
            relevant_keys_to_affmat.extend(with_prefix(get_keys(self.noiseConfig()),NOISE_PREFIX) )
            relevant_keys_to_affmat.extend(with_prefix(['id'],GENERAL_PREFIX))
        return relevant_keys_to_affmat
    def get_all_configs(self):
        """Gets the configuration for every  experiment. The corresponding prefix is added for each stage.
            Returns:
            `List[dict]` A list of all possible configs. 
        """
        g = self.generalConfig()
        for x in g:
            x["spec_name"] = self.get_spec_name()
        Z = [(g,GENERAL_PREFIX),
             (self.inputConfig(),INPUT_PREFIX),
             (self.noiseConfig(),NOISE_PREFIX),
             (self.filterConfig(),FILTER_PREFIX),
             (self.affmatConfig(),AFFMAT_PREFIX),
             (self.algConfig(),ALG_PREFIX)\
             ]
        l = [[spec.add_key_prefix(y, elem) for elem in x] for x,y in Z]        
        res =  list(reduce(lambda x,y: spec.comb(x,y),l))
        
         
        
        res = self.remove_undesirable_configs(res)
        LOG.info("Number of configurations: {}".format(len(res)))
        
        if self.CACHE_AFFMAT:
            res_df = pd.DataFrame(res)
            _cache_col = AFFMAT_PREFIX+"cache_dir"
            res_df[_cache_col] = '/tmp/{}'.format(self.get_spec_name())
            
            relevant_keys = self.get_keys_relevant_to_affmat()
            class Counter():
                def __init__(self):
                    self.counter = 0
                def inc(self):
                    self.counter += 1
                def value(self):
                    return self.counter
            COUNTER = Counter()
                
            def add_cache(df):
                
                df[_cache_col] = [os.path.join(x,str(COUNTER.value())) for x in df[_cache_col].values] 
                COUNTER.inc()
                print(COUNTER.value())
                return df
            
            res_df = res_df.fillna('__nan')
            for g, df in res_df.groupby(relevant_keys):
                print(g)
            res_df = res_df.groupby(relevant_keys).apply(add_cache)
            res_df = res_df.sort_values(axis=0,by=[_cache_col,GENERAL_PREFIX+'id'])
            """    
            ----------------------------------------
                Cache Directory handling
            -----------------------------------------
            """
            if self.CLEAN_AFFMAT_DIR:
                for _dir in pd.unique(res_df[_cache_col]):
                    print(_dir)
                    if os.path.isdir(_dir):
                        shutil.rmtree(_dir)
            print(list(res_df.index))
            res = list(np.array(res)[list(res_df.index)])
            
            _caches = res_df[_cache_col].values
            for i,dct in enumerate(res):
                dct[_cache_col] = _caches[i]
        
        return res
    
    
    
    @processify
    def run(self,cfg):
        from inspect import isclass
        exp =  self.experimentrun_type(cfg)
        res = exp.run(hook_list=TIME_HOOKS)
        del exp
        
        for x,y in zip([INPUT_PREFIX+'dataset',FILTER_PREFIX + 'filter', ALG_PREFIX + 'algorithm'],
                       [GSSLDataset,GSSLFilter,GSSLClassifier]):

            if isclass(cfg[x]) and issubclass(cfg[x],y):
                cfg[x] = cfg[x].get_name(cfg[x])

        
        
        res.update(cfg)
        return res
       
    
    def _append_to_csv(self,output_dicts,result_path,f_mode,cfgs_keys):
        all_keys = set().union(*(d.keys() for d in output_dicts))
        
        for k in all_keys:
            cfgs_keys.add(k)
        
        df = pd.DataFrame(index=range(len(output_dicts)), columns=np.sort(list(cfgs_keys)))
        for i in range(len(output_dicts)):
            x = output_dicts[i]
            for k in x.keys():
                df[k].iloc[i] = x[k]
        if not os.path.isdir(os.path.dirname(result_path)):
            os.mkdir(os.path.dirname(result_path))
        with open(result_path, f_mode) as f:            
                is_header = (f_mode == "w")
                df.to_csv(f, header = is_header)
                
    def aggregate_csv(self):
         
        CSV_PATH = os.path.join(CSV_FOLDER, self.get_spec_name() + '.csv')
        JOINED_CSV_PATH  = os.path.join(CSV_FOLDER, self.get_spec_name() + '_joined.csv')
        
        aggregate_csv([CSV_PATH], JOINED_CSV_PATH)
    
    def run_all(self):
        
        CSV_PATH = os.path.join(CSV_FOLDER, self.get_spec_name() + '.csv')
        JOINED_CSV_PATH  = os.path.join(CSV_FOLDER, self.get_spec_name() + '_joined.csv')
        
        cfgs = self.get_all_configs()
        cfgs_keys = set()
        for x in cfgs:
            cfgs_keys.update(x.keys())
        
        #List of produced output dicts
        output_dicts = list()
        
        cfgs_size = len(cfgs)
        
        has_written_already = False
        
        bar = progressbar.ProgressBar(maxval=cfgs_size)
        counter = 0
        bar.start()
        bar.update(0)
        
        for i in range(cfgs_size):
            print("PROGRESS: {}".format(i/cfgs_size))
            #Maybe suppress output
            nullwrite = open(os.devnull, 'w')   
            oldstdout = sys.stdout
            if not self.DEBUG_MODE:
                sys.stdout = nullwrite 

            output_dicts.append(self.run(cfgs[i]))
            
            sys.stdout = oldstdout
            #Append to csv if conditions are met
            if i == cfgs_size-1 or i % self.WRITE_FREQ == 0:
                LOG.info("appending csv...",LOG.ll.SPECIFICATION)
                csv_exists = os.path.isfile(CSV_PATH)
                if self.OVERWRITE:
                    if csv_exists and has_written_already:
                        f_mode = 'a'
                    else:
                        f_mode = 'w'
                else:
                    if csv_exists:
                        f_mode = 'a'
                    else:
                        f_mode = 'w'
                LOG.debug("f_mode={}".format(f_mode),LOG.ll.SPECIFICATION)
                self._append_to_csv(output_dicts,CSV_PATH,f_mode,cfgs_keys)
                has_written_already = True
                output_dicts.clear()
            
            bar.update(i+1)
        LOG.info(f"CSV saved at f{CSV_PATH}",LOG.ll.SPECIFICATION)
        aggregate_csv([CSV_PATH], JOINED_CSV_PATH)



    
    
    
    
