from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .fast_laddernet_se import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'pspnet': get_psp,
        'encnet': get_encnet,
        'shelfnet':get_laddernet,
    }
    return models[name.lower()](**kwargs)
