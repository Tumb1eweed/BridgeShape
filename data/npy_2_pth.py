import glob
import sys
import os
import numpy as np
import os.path as osp
import torch

if __name__ == '__main__':
    data_root = os.environ.get("DATA_ROOT", "/root/autodl-tmp/datasets")
    sdf_path = osp.join(data_root, "shapenet_dim32_sdf_npy")
    df_path = osp.join(data_root, "shapenet_dim32_df_npy")
    out_path = osp.join(data_root, "control_data")
    clss = ['02933112', '04530566', '03636649', '02691156', '02958343', '04379243', '04256520', '03001627']

    for cls in clss:
        sdfs = osp.join(sdf_path, cls)
        sdf_files = os.listdir(sdfs)
        for sdf_file in sdf_files:
            sdf_name = sdf_file[:-4]
            gt_file = sdf_name[:-3] + '0__.npy'
            sdf = np.load(osp.join(sdf_path, cls, sdf_file))
            df = np.load(osp.join(df_path, cls, gt_file))
            out_cls_path = osp.join(out_path, cls)
            os.makedirs(out_cls_path, exist_ok=True)
            out_file = osp.join(out_cls_path, sdf_name + '.pth')
            torch.save((sdf, df), out_file)
