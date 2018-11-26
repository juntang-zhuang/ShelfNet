from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .enc_laddernet import *
from .fast_laddernet_se import *
from .fcn_laddernet import   *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'pspnet': get_psp,
        'encnet': get_encnet,
        'laddernet':get_laddernet,
        'encladdernet':get_encladdernet,
        'fcnladdernet':get_fcnladder
    }
    return models[name.lower()](**kwargs)
