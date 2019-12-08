import os
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import time
import os
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

from shelfnet import ShelfNet
#from official_model_speed import BiSeNet
#from model import BiSeNet
def test():
    model = ShelfNet(n_classes=19)

    # official implementation
    #model = BiSeNet(19,criterion=None,ohem_criterion=None,is_training=False)

    print(model)
    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %.3fM" % (pytorch_total_params/1e6))

    model = model.cuda()
    model.eval()

    run_time = list()

    for i in range(0,100):
        input = torch.randn(1,3,1024,2048).cuda()
        # ensure that context initialization and normal_() operations
        # finish before you start measuring time
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            #output = model(input)#, aucx=False)
            output = model(input , aux=False)

        torch.cuda.synchronize()  # wait for mm to finish
        end = time.perf_counter()

        print(end-start)

        run_time.append(end-start)

    run_time.pop(0)

    print('Mean running time is ', np.mean(run_time))

if __name__ == "__main__":
    #args = Options().parse()

    test()
