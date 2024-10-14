# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import json, argparse, os
from tqdm import tqdm
from pathlib import Path
from src.scripts import optim_perpixel, optim_ganlatent, render_maps, render_for_loss

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', type=str, default='random')
    parser.add_argument('--json_path', type=str, default='./aigc_data.json')
    parser.add_argument('--result_path', type=str, default='/root/test_data/aigc/')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    # parser.add_argument('--input_num', type=int, default=1)
    args = parser.parse_args()
    
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    test_type = args.test_type
    if test_type not in ['random', 'uniform', 'drl', 'focus']:
        print("[ERR] Test type: random, uniform, drl, focus")
        exit(1)
    json_path = args.json_path
    
    result_path = args.result_path
    if result_path.endswith('/'):
        print("[WARN] Remove trailing slash from result path")
        result_path = result_path[:-1]
    result_path = Path(result_path + "/" + test_type)
    result_path.mkdir(parents=True, exist_ok=True)
    
    # begin generation    
    with open(json_path, 'r') as f:
        paths = json.load(f)['paths']
    
    # paths = paths[:10]
    
    test_range = 4 if test_type == 'focus' else 10
    
    for path in tqdm(sorted(paths)[:1000]):
        for i in range(test_range, 0, -1):
            print(f"Processing: {path}")
            material_dir = Path(path)
            material_name = path.split('/')[-1]
            save_dir = result_path / material_name
            json_name = material_name + '_' + test_type +'.json'
            json_dir = Path(path + "/" + json_name)
            #debug
            # print("debug: material_dir: {}, save_dir: {}, json_name: {}, json_dir: {}", material_dir, save_dir, json_name, json_dir)

            if not material_dir.exists():
                print("[WARN] Material directory not found: {}", material_dir)
                continue
            
            # Optimize texture maps by MaterialGAN for resolution 256x256
            # Different initialization
            # ckp = "auto"  # automatically choose initial ckp
            # ckp = ["ckp/latent_const_W+_256.pt", "ckp/latent_const_N_256.pt"]  # embeded latent and noise for constant maps (lower roughness)
            ckp = ["ckp/latent_avg_W+_256.pt"]  # average latent and zero noise
            # render_maps(json_dir, material_name, ["", "target", "", "", str(save_dir)], 256, 0.02, [1000, 10, 10], ckp)
            # render_for_loss(json_dir, Path("/root/results/materialgan/multi-pics_noiseless-2/random/acg_facade_020_a_3/10_tex.png"), ["", "target", "", "", str(save_dir)])
            optim_ganlatent(json_dir, material_name, ["", "target", "", "", str(save_dir)], 256, 0.02, [2000, 10, 10], ckp, i)