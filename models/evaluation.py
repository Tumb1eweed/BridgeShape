import os
from typing import List, Optional, Tuple
import torch.distributed as distr
import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import pytorch3d
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
import h5py
import random
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=''):
        self.reset()
        # name is the name of the quantity that we want to record, used as tag in tensorboard
        self.name = name
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1, summary_writer=None, global_step=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not summary_writer is None:
            # record the val in tensorboard
            summary_writer.add_scalar(self.name, val, global_step=global_step)


class Evaluator(object):
    def __init__(
        self,
        output_pcl_dir,
        dataset_root,
        dataset,
        summary_dir,
        experiment_name,
        device="cuda",
        res_gts="8192_poisson",
    ):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.summary_dir = summary_dir
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, "pointclouds", "test", res_gts)
        self.gts_mesh_dir = os.path.join(dataset_root, dataset, "meshes", "test")
        self.res_gts = res_gts
        self.device = device
        self.load_data()

    def load_data(self):
        self.pcls_up = load_xyz(self.output_pcl_dir)
        self.pcls_high = load_xyz(self.gts_pcl_dir)
        self.meshes = load_off(self.gts_mesh_dir)
        self.pcls_name = list(self.pcls_up.keys())

    def run(self):
        pcls_up, pcls_high, pcls_name = self.pcls_up, self.pcls_high, self.pcls_name
        results = {}

        for name in tqdm(pcls_name, desc="Evaluate"):
            pcl_up = pcls_up[name]
            if pcl_up.dim() != 2:
                continue

            pcl_up = pcl_up[:, :3].unsqueeze(0).to(self.device)

            if name not in pcls_high:
                self.logger.warning("Shape `%s` not found, ignored." % name)
                continue
            pcl_high = pcls_high[name].unsqueeze(0).to(self.device)
            verts = self.meshes[name]["verts"].to(self.device)
            faces = self.meshes[name]["faces"].to(self.device)

            cd_sph = chamfer_distance_unit_sphere(pcl_up, pcl_high)[0].item()

            if "blensor" in self.experiment_name:
                rotmat = torch.FloatTensor(Rotation.from_euler("xyz", [-90, 0, 0], degrees=True).as_matrix()).to(
                    pcl_up[0]
                )
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_up[0].matmul(rotmat.t()), verts=verts, faces=faces
                ).item()
            else:
                p2f = point_mesh_bidir_distance_single_unit_sphere(pcl=pcl_up[0], verts=verts, faces=faces).item()

            results[name] = {
                "cd_sph": cd_sph,
                "p2f": p2f,
            }

        results = pd.DataFrame(results).transpose()
        res_mean = results.mean(axis=0)
        logger.info("\n" + repr(results))
        logger.info("\nMean\n" + "\n".join(["%s\t%.12f" % (k, v) for k, v in res_mean.items()]))

        update_summary(
            os.path.join(self.summary_dir, "Summary_%s.csv" % self.dataset),
            model=self.experiment_name,
            metrics={
                # 'cd(mean)': res_mean['cd'],
                "cd_sph(mean)": res_mean["cd_sph"],
                "p2f(mean)": res_mean["p2f"],
                # 'hd_sph(mean)': res_mean['hd_sph'],
            },
        )


def update_summary(path, model, metrics):
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, sep="\s*,\s*", engine="python")
    else:
        df = pd.DataFrame()
    for metric, value in metrics.items():
        setting = metric
        if setting not in df.columns:
            df[setting] = np.nan
        df.loc[model, setting] = value
    df.to_csv(path, float_format="%.12f")
    return df

from .loss import l1
import mcubes

def evaluate_sdf(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: DictConfig,
    save_result_path:str=None,
    visualize: bool = False,
    step:int=0,
) -> dict:
    if  visualize:
        out_dir=os.path.join(save_result_path,'visualize')
    
    print("cfg.data.trunc_distance",cfg.data.trunc_distance)
    num_iter_per_epoch = len(val_loader)
    pbar = tqdm(enumerate(val_loader), total=num_iter_per_epoch)
    L1_meter = AverageMeter()
    loss_all=0.0
    loss_count=0
    for idx, eval_data in pbar:
        #if cfg.mvp_dataset_config.dataset == 'shapenet' or cfg.mvp_dataset_config.dataset == 'scannet':
        #    input_sdf, gt_df, bbox, _ = eval_data
        scan_ids, input_sdf, gt_df = eval_data
        input_sdf = input_sdf.cuda().detach() if input_sdf is not None else None #[B,2,32,32,32]
        gt_df = gt_df.cuda().detach() if gt_df is not None else None             #[B,1,32,32,32]

        with torch.no_grad():
            model_out,pred_x0= model.latent_sample( #[],[B,1,32,32,32]
                x1=input_sdf,
            )
        
        batch = gt_df.shape[0]
        pred_x0 = torch.clip(pred_x0, 0, cfg.data.trunc_distance)
        l1_loss=l1(pred_x0[:, 0], gt_df).mean()
        L1_meter.update((l1_loss).detach(), n=batch)  # l1
        
        l1_error=torch.abs(pred_x0[:, 0]- gt_df)
        loss_all+= l1_error.view(l1_error.size(0), -1).mean(dim=1).sum(0)
        
        loss_count+=batch
        
        if visualize:
            # # Visualize range scans (by SDF)
            for i in range(len(input_sdf)):
                single_observe = input_sdf[i]
                obs_sdf = single_observe[0].cpu().numpy()
                scan_id = scan_ids[i]
                sdf_vertices, sdf_traingles = mcubes.marching_cubes(obs_sdf, 0.5)
                out_file = os.path.join(out_dir,"obj", f'{scan_id}input.obj')
                obj_dir =os.path.dirname(out_file)
                if os.path.exists(obj_dir)==False:
                    os.makedirs(obj_dir)
                mcubes.export_obj(sdf_vertices, sdf_traingles, out_file)
                # print(f"Save {out_file}!")

            # Visualize GT DF
            for i in range(len(gt_df)):
                gt_single = gt_df[i].cpu().numpy()
                scan_id = scan_ids[i]
                vertices, traingles = mcubes.marching_cubes(gt_single, 0.5)
            #    # vertices = (vertices.astype(np.float32) - 0.5) / config.exp.res - 0.5
                out_file = os.path.join(out_dir,"obj", f'{scan_id}gt.obj')
                obj_dir =os.path.dirname(out_file)
                if os.path.exists(obj_dir)==False:
                    os.makedirs(obj_dir)
                mcubes.export_obj(vertices, traingles, out_file)
                # print(f"Save {out_file}!")

            pred_x0 = pred_x0.cpu().numpy()[:, 0]

            # Visualize predicted DF
            for i in range(len(pred_x0)):
                low_sample = pred_x0[i]
                scan_id = scan_ids[i]
                # You can choose more advanced surface extraining methods for TDF outputs
                vertices, traingles = mcubes.marching_cubes(low_sample, 0.5)
                out_file = os.path.join(out_dir,"obj", f'{scan_id}pred.obj')
                mcubes.export_obj(vertices, traingles, out_file)
                #out_npy_file = os.path.join(out_dir,'npy', f'{scan_id}pred.npy')
                #obj_npy =os.path.dirname(out_npy_file)
                #if os.path.exists(obj_npy)==False:
                #    os.makedirs(obj_npy)
                #np.save(out_npy_file, low_sample)
    l1_loss  = L1_meter.avg
    result = 'Step {}, '.format(step) + \
              'Epoch {}, '.format(step//cfg.training.log_interval) + \
              'l1_error: {:.4f}, '.format(l1_loss)

    print(result)
    print(loss_all/loss_count)
    dir=os.path.join(cfg.save_dir,'%d_steps'% cfg.diffusion.sampling_timesteps)
    if not os.path.exists(dir):
        os.makedirs(dir)
    output_file = os.path.join(dir, 'result.txt')
    with open(output_file, 'a') as f:
        f.write(result + '\n')


import re

def extract_mesh(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: DictConfig,
    save_result_path:str=None,
    visualize: bool = False,
    step:int=0,
) -> dict:
    out_dir="/workspace/EPN/control_data/shapenet_dim64_df_8c_npy/mesh/04530566"
    num_iter_per_epoch = len(val_loader)
    pbar = tqdm(enumerate(val_loader), total=num_iter_per_epoch)
    print("cfg.data.trunc_distance:",cfg.data.trunc_distance)

    for idx, eval_data in pbar:
        scan_ids, input_sdf, gt_df = eval_data
        gt_df = gt_df.cuda().detach() if gt_df is not None else None             #[B,1,32,32,32]

        for i in range(len(gt_df)):
            gt_single = gt_df[i].cpu().numpy()
            scan_id = scan_ids[i]
            vertices, traingles = mcubes.marching_cubes(gt_single, 0.5)
            out_file = os.path.join(out_dir,"obj", f'{scan_id}gt.obj')
            out_file = re.sub(r'__\d{1}__', '', out_file)
            if os.path.exists(out_file):
                continue
            obj_dir = os.path.dirname(out_file)
            if os.path.exists(obj_dir)==False:
                os.makedirs(obj_dir)
            mcubes.export_obj(vertices, traingles, out_file)
            print(f"Save {out_file}!")


