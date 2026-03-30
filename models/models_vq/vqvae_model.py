import os
from collections import OrderedDict

import numpy as np
import mcubes
import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.profiler import record_function

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.models_vq.base_model import BaseModel
from models.models_vq.networks.vqvae_networks.network import VQVAE
from models.models_vq.losses import VQLoss
from models.loss import l1
import utils.utils_vq.util
from utils.utils_vq.util_3d import init_mesh_renderer, render_sdf
from utils.utils_vq.distributed import reduce_loss_dict
import mcubes
def v(gt_df, pred_df):
    print(gt_df.shape)
    print(pred_df.shape)
    out_dir="v"
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    gt_df = gt_df[:, 0]
    j = torch.randint(low=0, high=1000, size=(1,)).item()
    for i in range(len(gt_df)):
        gt_single = gt_df[i].cpu().numpy()
        vertices, traingles = mcubes.marching_cubes(gt_single, 0.5)
        # vertices = (vertices.astype(np.float32) - 0.5) / config.exp.res - 0.5
        out_file = os.path.join(out_dir, "obj", f'{j}gt.obj')
        obj_dir =os.path.dirname(out_file)
        if os.path.exists(obj_dir) == False:
            os.makedirs(obj_dir)
        mcubes.export_obj(vertices, traingles, out_file)
        # print(f"Save {out_file}!")

    pred_df = pred_df.cpu().numpy()[:, 0]

    # Visualize predicted DF
    for i in range(len(pred_df)):
        low_sample = pred_df[i]
        vertices, traingles = mcubes.marching_cubes(low_sample, 0.5)
        out_file = os.path.join(out_dir, "obj", f'{j}pred.obj')
        mcubes.export_obj(vertices, traingles, out_file)

class VQVAEModel(BaseModel):
    def name(self):
        return 'VQVAE-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.trunc_thres=opt.trunc_thres

        # -------------------------------
        # Define Networks
        # -------------------------------

        # model
        assert opt.vq_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig

        self.vqvae = VQVAE(ddconfig, n_embed, embed_dim)
        self.vqvae.to(self.device)

        if self.isTrain:
            # define loss functions
            codebook_weight = configs.lossconfig.params.codebook_weight
            self.loss_vq = VQLoss(codebook_weight=codebook_weight).to(self.device)

            # initialize optimizers
            self.optimizer = optim.Adam(self.vqvae.parameters(), lr=opt.lr, betas=(0.5, 0.9))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        # continue training
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)

        if opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 20
        else:
            dist, elev, azim = 1.7, 20, 20
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for saving best ckpt
        self.best_iou = 1e12

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.vqvae_module = self.vqvae.module
        else:
            self.vqvae_module = self.vqvae

    def switch_eval(self):
        self.vqvae.eval()
        
    def switch_train(self):
        self.vqvae.train()

    def make_distributed(self, opt):
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

    def set_input(self, input):

        x = input[2].unsqueeze(1) #[1,res,res,res]
        #print(x.shape)
        # print("input_x_shape: ",x.shape)
        self.x = x
        self.cur_bs = x.shape[0] # to handle last batch
        vars_list = ['x']

        self.tocuda(var_names=vars_list)

    def forward(self):
        self.x_recon, self.qloss = self.vqvae(self.x, verbose=False)
        # print(self.x_recon.shape,self.qloss)
        # print("-----------------------------------------")

    @torch.no_grad()
    def inference(self, data, should_render=False, verbose=False):
        self.switch_eval()
        self.set_input(data)

        with torch.no_grad():
            self.z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)
            self.x_recon = self.vqvae_module.decode_no_quant(self.z)
            self.x_recon = torch.clamp(self.x_recon, min=0, max=self.trunc_thres)
            if should_render:
                self.image = render_sdf(self.renderer, self.x)
                self.image_recon = render_sdf(self.renderer, self.x_recon)

        self.switch_train()

    def test_iou(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        self.inference(data, should_render=False)

        x = self.x
        x_recon = self.x_recon
        iou = utils.utils_vq.util.iou(x, x_recon, thres)

        return iou
    
    def test_l1(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        self.inference(data, should_render=False)

        x = self.x
        x_recon = self.x_recon
        l1_loss=l1(x_recon[:, 0], x[:,0])
        return l1_loss

    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        # self.eval()
        self.switch_eval()
        
        l1_list=[]
        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                
                l1 = self.test_l1(test_data)
                l1_list.append(l1.detach())
                iou = self.test_iou(test_data, thres=0.5)
                iou_list.append(iou.detach())

        l1 = torch.cat(l1_list)
        l1_mean, l1_std = l1.mean(), l1.std()
        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()
        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
            ('l1', l1_mean.data),
            ('l1_std', l1_std.data),
        ])
        if global_step == -1:
            return ret
        # check whether to save best epoch
        if ret['l1'] < self.best_iou:
            self.best_iou = ret['l1']
            save_name = f'epoch-best'
            self.save(save_name, global_step) # pass 0 just now

        self.switch_train()
        return ret


    def backward(self):
        '''backward pass for the generator in training the unsupervised model'''
        total_loss, loss_dict = self.loss_vq(self.qloss, self.x, self.x_recon)

        self.loss = total_loss

        self.loss_dict = reduce_loss_dict(loss_dict)

        self.loss_total = loss_dict['loss_total']
        self.loss_codebook = loss_dict['loss_codebook']
        self.loss_nll = loss_dict['loss_nll']
        self.loss_rec = loss_dict['loss_rec']

        self.loss.backward()

    def optimize_parameters(self, total_steps):

        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
    
    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.mean().data),
            ('codebook', self.loss_codebook.mean().data),
            ('nll', self.loss_nll.mean().data),
            ('rec', self.loss_rec.mean().data),
        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.image = render_sdf(self.renderer, self.x)
            self.image_recon = render_sdf(self.renderer, self.x_recon)

        vis_tensor_names = [
            'image',
            'image_recon',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)

    def save(self, label, global_step=0, save_opt=False):

        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_step,
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'vqvae_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)

    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.cpu().state_dict()
        self.vqvae.quantize.embedding.cuda()
        return ret

    def load_ckpt(self, ckpt, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        # NOTE: handle version difference...
        if 'vqvae' not in state_dict:
            self.vqvae.load_state_dict(state_dict)
        else:
            self.vqvae.load_state_dict(state_dict['vqvae'])
            
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


