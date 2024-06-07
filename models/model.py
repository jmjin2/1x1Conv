from collections import OrderedDict
import torch.nn as nn
from torch.optim import Adam

from models.base_model import BaseModel
from arch.FusionA_arch import FusionA
from arch.FusionB_arch import FusionB
from utils.util import CharbonnierLoss

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)

        # define network
        if opt.model == "FusionA":
            self.net_g = FusionA(load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
        elif opt.model == "FusionB":
            self.net_g = FusionB(load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
        # self.print_network(self.net_g)
        load_path = self.opt.pretrained_path
        if load_path is not None:
            self.load_network(self.net_g, load_path, True, 'params')

        if self.is_train:
            self.init_training_settings(opt)

    def init_training_settings(self, opt):
        self.net_g.train()

        self.ema_decay = 0.999
        if self.ema_decay > 0:
            if opt.model == "FusionA":
                self.net_g_ema = FusionA(load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
            elif opt.model == "FusionB":
                self.net_g_ema = FusionB(load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
            
            # load pretrained model
            load_path = self.opt.pretrained_path
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, True, 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.cri_pix = CharbonnierLoss().to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []
        excluded_params = set(self.net_g.basicvsr.parameters())
        for k, v in self.net_g.named_parameters():
            if v.requires_grad and k not in excluded_params:
                optim_params.append(v)

        self.optimizer_g = Adam(optim_params, lr=1e-4,
                                    betas=[0.9,0.99])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq1 = data['lq1'].to(self.device)
        self.lq2 = data['lq2'].to(self.device)
        self.lq3 = data['lq3'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)



# optimize parameters
    def optimize_parameters(self, current_step):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq1, self.lq2, self.lq3)
        l_total = 0
        l_pix = self.cri_pix(self.output, self.gt)
        print(l_pix)

        l_total += l_pix
        l_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def current_loss(self):
        return self.l_pix
    
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)