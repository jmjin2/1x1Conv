from collections import OrderedDict
import torch.nn as nn
from torch.optim import Adam

from models.base_model import BaseModel
from arch.multiview_arch import MultiViewSR
from arch.multiviewSkip_arch import MultiViewSkipSR

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)

        # define network
        if opt.model == "MultiviewSR":
            self.net_g = MultiViewSR(num_feat=opt.num_feat, num_block=opt.num_block, load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
        elif opt.model == "MultiviewSkipSR":
            self.net_g = MultiViewSkipSR(num_feat=opt.num_feat, num_block=opt.num_block, load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
        self.print_network(self.net_g)

        if self.is_train:
            self.init_training_settings(opt)

    def init_training_settings(self, opt):
        self.net_g.train()

        self.ema_decay = 0.999
        if self.ema_decay > 0:
            if opt.model == "MultiviewSR":
                self.net_g_ema = MultiViewSR(num_feat=opt.num_feat, num_block=opt.num_block, load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
            elif opt.model == "MultiviewSkipSR":
                self.net_g_ema = MultiViewSkipSR(num_feat=opt.num_feat, num_block=opt.num_block, load_path=opt.basicvsr_path, spynet_path=None).to(self.device)
            # load pretrained model
            load_path = self.opt.pretrained_path
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.cri_pix = nn.MSELoss().to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
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
        # print(self.output.shape)
        # print(self.gt.shape)
        l_pix = self.cri_pix(self.output, self.gt)
        print(l_pix)

        l_total += l_pix
        l_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

# # Test
#     def test(self):
#         self.netG.eval()
#         with torch.no_grad():
#             self.netG_forward()
#         self.netG.train()

    def current_loss(self):
        return self.l_pix

#     def current_visuals(self, need_H=True):
#         out_dict = OrderedDict()
#         out_dict['L'] = self.L.detach()[0].float().cpu()
#         out_dict['E'] = self.E.detach()[0].float().cpu()
#         if need_H:
#             out_dict['H'] = self.H.detach()[0].float().cpu()
#         return out_dict

#     def current_results(self, need_H=True):
#         out_dict = OrderedDict()
#         out_dict['L'] = self.L.detach().float().cpu()
#         out_dict['E'] = self.E.detach().float().cpu()
#         if need_H:
#             out_dict['H'] = self.H.detach().float().cpu()
#         return out_dict


#     def print_network(self):
#         msg = self.describe_network(self.netG)
#         print(msg)

#     def print_params(self):
#         msg = self.describe_params(self.netG)
#         print(msg)

#     def info_network(self):
#         msg = self.describe_network(self.netG)
#         return msg

#     def info_params(self):
#         msg = self.describe_params(self.netG)
#         return msg
    
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)