import torch
import numpy as np
from collections.abc import Iterable
from sklearn.metrics.pairwise import cosine_similarity
import logging, os, sys

from torch import Tensor


class LabelSmoothing(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, Iterable):
            val = [val]

        val = np.asarray(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch >= np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        idx = torch.from_numpy(np.asarray(topk) - 1)
        return correct.cumsum(0).sum(1)[idx] * 100.0 / batch_size

def to_cuda_maybe(obj):
    if torch.cuda.is_available():
        if isinstance(obj, Iterable) and (not isinstance(obj, Tensor)):
            return [e.cuda() for e in obj]
        else:
            return obj.cuda()
    return obj


def freeze_bn(module):
    for child in module.modules():
        if isinstance(child, torch.nn.BatchNorm2d):
            child.eval()


def similarity(cuda_arr):
    tmp = cuda_arr.detach().cpu().t().numpy()
    print(cosine_similarity(tmp))
    print("------------------")


def cuda_to_np(arr, dtype=np.float64):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(dtype)
    return arr.astype(dtype)


def to_np_maybe(obj):
    if isinstance(obj, Iterable):
        return [cuda_to_np(e) for e in obj]
    return cuda_to_np(obj)


def print_metrics(names, vals):
    ret = ""
    for key, val in zip(names, vals):
        ret += f"{key}: {val}  "
    return ret


def np_to_cuda(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.astype(np.float32)).cuda()

    return arr


def partial_reload(model, state_dict):
    cur_dict = model.state_dict()
    partial_dict = {}
    for k, v in state_dict.items():
        if k in cur_dict and cur_dict[k].shape == v.shape:
            partial_dict[k] = v
    cur_dict.update(partial_dict)
    model.load_state_dict(cur_dict)


def get_logger(name, dir="logs/", file_name=None, log_level=logging.INFO):
    #local machine, no file output
    logger = logging.getLogger(name)

    c_handler = logging.StreamHandler(stream=sys.stdout)
    c_handler.setLevel(log_level)
    c_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    file_path = os.path.join(dir, file_name)
    f_handler = logging.FileHandler(file_path)
    # f_handler.setLevel(log_level)
    f_format = logging.Formatter('%(asctime)s | %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.setLevel(log_level)
    return logger


def save_routine(epoch, model, optimizer, save_path):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
    }
    torch.save(state, save_path)











