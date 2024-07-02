import torch
import yaml
import os
import matplotlib.pyplot as plt
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def load_yaml(yaml_path:str = 'configs.yaml'):
    """ 读配置文件, 默认configs.yaml """
    with open(yaml_path, 'r', encoding='UTF-8') as f:
        data = yaml.safe_load(f)
    return data

def load_state_dict(path, model,
                    optim=None,
                    scheduler=None):
    """ 读取 model state dict """
    state = torch.load(path, map_location="cpu")
    # if not is_distributed:
    model.load_state_dict(state['module'])
    # else:
    #     new_state_dict = OrderedDict()
    #     for k, v in state.items():
    #         name = k[7:] # module字段在最前面，从第7个字符开始就可以去掉module
    #         new_state_dict[name] = v # 新字典的key值对应的value一一对应
    #     model.state_dict = new_state_dict
    min_loss = state['min_loss']
    epoch = state['epoch']
    cfgs = state['args']
    if optim is not None:
        optim.load_state_dict(state['optim'])
    if scheduler is not None:
       scheduler.load_state_dict(state['scheduler'])
    return model, cfgs, epoch, min_loss, optim, scheduler

def show_yaml(trace=print, args=None):
    """ 终端打印 """
    trace("Configureations:")
    trace('-'*55)
    trace('|%20s | %30s|' % ('keys', 'values'))
    trace('-'*55)
    for k, v in args.items():
        if isinstance(v, dict):
            for key, value in v.items():
                trace('|%20s | %30s' % (str(key), str(value)))
        else:
            trace('|%20s | %30s' % (str(k), str(v)))
    trace('-'*55)

def build_save_dir(save_dir='weights'):
    """ 新建文件保存模型权重 """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = os.listdir(save_dir)
    num = [int(f.split('_')[-1]) for f in files if f.find('run') >= 0]
    save_dir = os.path.join(save_dir, 'run_{}'.format(len(num) + 1))
    return save_dir

class TensorboardWriter:
    """ tensorboard记录训练中损失的变化情况 """
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = log_dir
        self.writer  = SummaryWriter(log_dir)


def get_optim_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_state_dict(path,
                    model,
                    cfgs=None,
                    max_acc=None,
                    min_loss=None,
                    epoch=None,
                    optim=None,
                    scheduler=None,
                    is_distributed=False):
    """ 保存 model state dict """
    if not is_distributed:
        state_dict = model.state_dict()
    else:
        state_dict = model.module.state_dict()
    state = {
        "module": state_dict,
        "max_acc": max_acc,
        "min_loss": min_loss,
        "epoch": epoch,
        "optim": optim.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": cfgs
    }
    torch.save(state, path)

class AverageMeter:
    def __init__(self):
        self.clean()
        self.lst = []

    def reset(self):
        super().__init__()

    def clean(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = 1e10
        self.max = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.avg > self.max:
            self.max = self.avg
        if self.avg < self.min:
            self.min = self.avg

    def append(self):
        self.lst.append(self.avg)


def plot_samples(samples, labels):
    A=1


def sim_matrix2(ori_vector, arg_vector, temp=1.0):
    for i in range(len(ori_vector)):
        sim = torch.cosine_similarity(ori_vector[i].unsqueeze(0), arg_vector, dim=1) * (1 / temp)
        if i == 0:
            sim_tensor = sim.unsqueeze(0)
        else:
            sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
    return sim_tensor

def compute_diag_sum(tensor):
    num = len(tensor)
    diag_sum = 0
    for i in range(num):
        diag_sum += tensor[i][i]
    return diag_sum