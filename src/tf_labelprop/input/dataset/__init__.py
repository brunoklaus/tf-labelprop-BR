
import gzip
import os
import shutil
import ssl
import zipfile
import os.path as osp
import os.path as osp
from six.moves import urllib
import tf_labelprop
from tf_labelprop.settings import INPUT_FOLDER

def extract_gz(fpath,fpath_extracted):
    
    with gzip.open(f'{fpath}', 'rb') as f_in:
        with open(fpath_extracted, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_zip(fpath):
    with zipfile.ZipFile(fpath, 'r') as f:
        f.extractall(osp.dirname(fpath))

def download_url(url, folder):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    if osp.exists(path):
        LOG.warn(f'File already found at {path}',log_loc=LogLocation.INPUT)
        return path
        
    LOG.info(f'Downloading {url} to {path}',log_loc=LogLocation.INPUT)
    
    if not osp.isdir(folder):
        os.makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())
    
    LOG.info(f'Downloading {url} to {path}..Done!',log_loc=LogLocation.INPUT)
    return path

    

class GSSLDataset(object):
    """ Skeleton class for GSSL Datasets. Based on Dataset class of Pytorch Geometric."""


    def __init__(self,root = INPUT_FOLDER):
        self.root = root 
    delete_raw_dir = True
    
    @staticmethod
    def get_name(self):
        return "---"
        
    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]
    
    @property
    def raw_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = list(self.processed_file_names)
        return [osp.join(self.raw_dir, f) for f in files]
    
    
    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    
    _name = None
    @property
    def name(self):
        if self._name is None:
            if self.__class__ == GSSLDataset:
                self._name = "---"
            else:
                self._name = self.__class__.__name__
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
    
    

    def download(self):
        """ Downloads data to :attr:self.raw_dir. """
        raise NotImplementedError(f"Did not implement method to download dataset)")
    
    def process(self):
        """ Processes data and stores in :attr:self.processed_dir. """
        raise NotImplementedError(f"Did not implement method to process dataset)")
    
    
    
    
    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError("")
    
    
    def load(self):
        """ Returns the input data.
        
        Args:
            X (`NDArray[float].shape[N,D]`) : Input matrix of N instances of dimension D.
            W (`NDArray[float].shape[N,N]`): The affinity matrix encoding the weighted edges.
            Y (`NDArray[float].shape[N,C]`): The initial belief matrix
            hook (GSSLHook): Optional. A hook to execute extra operations (e.g. plots) during the algorithm
        
        Returns:
            `Union[None,NDArray[float].shape[N,D]]`: The Feature Matrix X.
            `Union[None,:class:`tf_labelprop.gssl.graph.gssl_affmat.AffMat`]`: The Affinity Matrix W.
            `NDArray[float].shape[N,C]]`: The Label Matrix Y.
        """
        if not all([os.path.isfile(p) for p in self.raw_paths]):
            if not osp.isdir(self.raw_dir):
                os.makedirs(self.raw_dir)
            self.download()
                
        if not all([os.path.isfile(p) for p in self.processed_paths]):  
            if not osp.isdir(self.processed_dir):
                os.makedirs(self.processed_dir)
            self.process()
        

########################################################################################################
from tf_labelprop.input.dataset._mnist import MNIST
from tf_labelprop.input.dataset._toy_ds import ChapelleDataset, ISOLET
from tf_labelprop.logging  import logger as LOG
from tf_labelprop.logging.logger import LogLocation



