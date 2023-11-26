from functools import partial
import torch
import numpy as np

def approx_v(T, p, frac):
    if frac < 1.0:
        n_elements = T.numel()
        n_sample = min(int(max(np.ceil(n_elements * frac), np.ceil(100 / p))), n_elements)
        n_top = int(np.ceil(n_sample * p))
        if n_elements == n_sample:
            i = 0
        else:
            i = np.random.randint(n_elements - n_sample)

        topk, _ = torch.topk(T.flatten()[i:i + n_sample], n_top)
        if topk[-1] == 0.0 or topk[-1] == T.max():
            return approx_v(T, p, 1.0)
    else:
        n_elements = T.numel()
        n_top = int(np.ceil(n_elements * p))
        if T.device.type == "mps":
            topk, _ = torch.topk(T.flatten().cpu().abs(), n_top)
            topk = topk.to("mps:0")
            _ = _.to("mps:0")
        else:
            topk, _ = torch.topk(T.flatten(), n_top)

    return topk[-1], topk


def topk(T, hp):
    '''
    "Deep Gradient Compression: Reducing the communication Bandwidth for Distributed Training, Lin et al."
    '''
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)

    if hp_["cr"] > 1.0:
        return T

    T_abs = torch.abs(T)

    v, _ = approx_v(T_abs, hp_["cr"], hp_["approx"])

    out = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(T.device))

    return out


def ht(T, hp):
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)
    T_abs = torch.abs(T)
    v = hp_["cr"]
    out = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(T.device))

    return out

def compression_function(compression_config):
    '''
    Returns a function that maps a tensor to a tensor of the same shape
    '''
    name = compression_config["method"]
    hp = compression_config["params"]
    return partial(globals()[name], hp=hp)


def none(T, hp):
    return T