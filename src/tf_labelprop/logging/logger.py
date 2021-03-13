'''
Created on 19 de mai de 2020

@author: klaus
'''
from enum import Enum  
import logging
import sys


class LogLocation(Enum):
    """
    A LogLocation is an identifier which locates where a logged message may come from.
    """

    OTHER = 0
    EXPERIMENT = 1
    CLASSIFIER = 2
    FILTER = 3
    MATRIX = 4
    NOISE = 6
    HOOK = 5
    SPECIFICATION = 6
    UTILS = 7
    OUTPUT = 8
    INPUT=9
class _LogType(Enum):
    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3

ll = LogLocation

""" Flag that may force Debug to be Disabled """
NO_DEBUG = True

DEFAULT_ALLOWED_DEBUG_LOCATIONS = [ll.EXPERIMENT,ll.CLASSIFIER,ll.NOISE,
                           ll.FILTER,ll.MATRIX]


class _CustomLogFilter(logging.Filter):
    def __init__(self,locs= DEFAULT_ALLOWED_DEBUG_LOCATIONS):
        self.locs = locs
    
    def set_allowed_debug_locations(self,locs):
        self.locs = locs
    
    def filter(self, record):
        
        if record.type == _LogType.DEBUG:
            if NO_DEBUG:
                return False 
            return (record.location in self.locs)
        else:
            return True
ll = LogLocation

_logfilter = _CustomLogFilter()   
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_LOG = logging.getLogger('GSSL_log')
_LOG.setLevel(logging.DEBUG)
_LOG.addHandler(_handler)
_LOG.addFilter(_logfilter)


__map_to_logf = {_LogType.DEBUG: _LOG.debug,
                 _LogType.ERROR: _LOG.error,
                 _LogType.WARN: _LOG.warn,
                 _LogType.INFO: _LOG.info,
                 }
def set_allowed_debug_locations(locs):
    """
        Sets which of the identifiers (:class:`log.logger.LogLocation`) should NOT have
        their debug messages suppressed.
        
        Args:
            locs (List[:class:`log.logger.LogLocation`])
    """
    _logfilter.set_allowed_debug_locations(locs)
         

def log(msg,log_type,log_loc=LogLocation.OTHER):
    """
        Logs data to this module's logger.
    """
    f =__map_to_logf[log_type]
    
            
    f(msg,extra={'location':log_loc,'type':log_type})
    
def info(msg,log_loc=LogLocation.OTHER):
    log(msg,_LogType.INFO,log_loc)

def debug(msg,log_loc=LogLocation.OTHER):
    log(msg,_LogType.DEBUG,log_loc)

def error(msg,log_loc=LogLocation.OTHER):
    log(msg,_LogType.ERROR,log_loc)

def warn(msg,log_loc=LogLocation.OTHER):
    log(msg,_LogType.WARN,log_loc)



    
