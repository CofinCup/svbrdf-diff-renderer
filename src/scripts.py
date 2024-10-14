# -*- coding: utf-8 -*-
#
# Copyright (c) 2024, Yu Guo. All rights reserved.

import os
import shutil
import numpy as np
import torch as th
from tqdm import tqdm

from pathlib import Path

from .capture import Capture
from .materialgan import MaterialGANOptim
from .microfacet import Microfacet
from .svbrdf import SvbrdfIO, SvbrdfOptim
from .mitsubarender import MitsubaRender
from .imageio import imread, imwrite, img2gif


def gen_textures_from_materialgan(json_dir):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, None, device)

    optim_obj = MaterialGANOptim(device, None, "ckp/materialgan.pth")
    optim_obj.init_from([])
    textures = optim_obj.latent_to_textures(optim_obj.latent)

    svbrdf_obj.save_textures_th(textures, svbrdf_obj.reference_dir)


def render(json_dir, dir5, res):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, dir5, device)
    textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir, res)

    render_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval(textures)

    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.target_dir)

def render_for_loss(mat_path, dir5):
    # epochs: list of 3 int. [0] is total epochs, [1] is epochs for latent, [2] is for noise, in each cycle.
    # tex_init: string. [], [PATH_TO_LATENT.pt], or [PATH_TO_LATENT.pt, PATH_TO_NOISE.pt]

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(Path('/root/code/svbrdf-diff-renderer/render0.json'), dir5, device)
    textures = svbrdf_obj.load_textures_th(mat_path)

    render_obj = Microfacet(256, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval_only_diffuse(textures.to(device))
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)
    
    svbrdf_obj = SvbrdfIO(Path('/root/code/svbrdf-diff-renderer/render1.json'), dir5, device)
    textures = svbrdf_obj.load_textures_th(mat_path)

    render_obj = Microfacet(256, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval_with_trick(textures.to(device))
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)

    svbrdf_obj = SvbrdfIO(Path('/root/code/svbrdf-diff-renderer/render2to9.json'), dir5, device)
    textures = svbrdf_obj.load_textures_th(mat_path)

    render_obj = Microfacet(256, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval(textures.to(device))
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)

def apply_transform(pos, angle_x, angle_y, angle_z):
    # 绕x轴旋转
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.radians(angle_x)), -np.sin(np.radians(angle_x))],
                   [0, np.sin(np.radians(angle_x)), np.cos(np.radians(angle_x))]])
    
    # 绕y轴旋转
    Ry = np.array([[np.cos(np.radians(angle_y)), 0, np.sin(np.radians(angle_y))],
                   [0, 1, 0],
                   [-np.sin(np.radians(angle_y)), 0, np.cos(np.radians(angle_y))]])
    
    # 绕z轴旋转
    Rz = np.array([[np.cos(np.radians(angle_z)), -np.sin(np.radians(angle_z)), 0],
                   [np.sin(np.radians(angle_z)), np.cos(np.radians(angle_z)), 0],
                   [0, 0, 1]])
    
    # 应用旋转矩阵，顺序是X -> Y -> Z
    pos = np.dot(Rx, np.dot(Ry, np.dot(Rz, pos)))
    
    return pos

def render_envmap(result_dir, map_path, res):
    parent_dir = os.path.dirname(map_path)
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    tmp_dir = os.path.join(parent_dir, 'tmp', map_name)
    
    if not os.path.exists(tmp_dir) :
        os.mkdir(tmp_dir)
    
    result_dir = result_dir / map_name
    
    vid_dir = result_dir / "vid"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # if not map_path.exists():
    #         raise FileNotFoundError(f"[ERROR:SvbrdfIO:load_textures_th] {map_path} does not exist")

    # # 读取 SVBRDF 拼接图
    # svbrdf = imread(map_path, "srgb")
    # height = svbrdf.shape[0]   # 1024
    # width = svbrdf.shape[1]    # 应该是 4096 (4 * 1024)
    # # 如果图片宽度不为4倍宽度，则读取后4个图片宽度
    # if width < 4 * height:
    #     raise ValueError(f"Expected width to be at least 4 times the height, but got {width}")
    # elif width > 4 * height:
    #     # 读取后4个图片宽度
    #     print(f"[WARNING:SvbrdfIO:load_textures_th] Detected extra width. Adjusting to the last 4 textures.")
    #     svbrdf = svbrdf[:, -4*height:]  # 保留最后4个贴图的部分
    # normal = svbrdf[:, :height]                    # 法线贴图
    # diffuse = svbrdf[:, height:2*height]           # 漫反射贴图
    # roughness = svbrdf[:, 2*height:3*height]       # 粗糙度贴图
    # specular = svbrdf[:, 3*height:]                # 镜面反射贴图
    # imwrite(normal, os.path.join(tmp_dir, f"nom.png"), flag="srgb")
    # imwrite(roughness, os.path.join(tmp_dir, f"rgh.png"), flag="srgb")
    # imwrite(specular, os.path.join(tmp_dir, f"spe.png"), flag="srgb")
    # imwrite(diffuse, os.path.join(tmp_dir, f"dif.png"), flag="srgb")

    # specular = imread(os.path.join(tmp_dir, f"spe.png"), "srgb")
    # specular = specular ** 2.2
    # imwrite(specular, os.path.join(tmp_dir, f"spe.png"), flag="srgb")

    render_obj = MitsubaRender([res, res], "fig/ennis.exr", str(tmp_dir))
    
    # 半球面上的旋转
    radius = 3
    theta = np.pi / 4  # 固定极角45度

    # 方位角范围：从0到2π
    angle_range = np.linspace(0, 2 * np.pi, num=60)

    for i, phi in tqdm(enumerate(angle_range)):
        if i not in range(19,23,1):
            continue
        # 根据球坐标公式转换为新的笛卡尔坐标
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        # 应用平面的旋转变换，得到相对新z轴的光源位置
        new_light_pos = apply_transform([x, y, z], -30, 0, -10)
        
        camera_pos = [0, 0, 1]

        # 更新光源位置
        render_obj.scene['light']['position'] = new_light_pos

        im = render_obj.render(new_light_pos, 45, camera_pos)
        imwrite(im, vid_dir / f"{i:03d}.png", flag="srgb")

    # for j, ii in enumerate(range(i-1, 0, -1)):
    #     shutil.copy(str(vid_dir / f"{ii:03d}.png"), str(vid_dir / f"{i+j+1:03d}.png"))

    # img2gif(vid_dir.glob("*.png"), result_dir / f"{map_name}.gif", method="Pillow")


def gen_targets_from_capture(data_dir, size=17.0, depth=0.1):
    input_obj = Capture(data_dir)
    input_obj.eval(size, depth)


def optim_perpixel(json_dir, dir5, res, lr, epochs, tex_init):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_dir, dir5, device)
    targets = svbrdf_obj.load_images_th(svbrdf_obj.target_dir, res)

    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    optim_obj = SvbrdfOptim(device, renderer_obj)
    optim_obj.load_targets(targets)

    if tex_init == "random":
        optim_obj.init_from_randn()
    elif tex_init == "const":
        optim_obj.init_from_const()
    elif tex_init == "textures":
        textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir, res)
        optim_obj.init_from_tex(textures)
    else:
        exit()

    optim_obj.optim(epochs, lr, svbrdf_obj)

    svbrdf_obj.save_textures_th(optim_obj.textures.clamp(-1, 1), svbrdf_obj.optimize_dir)
    rendereds = renderer_obj.eval(optim_obj.textures.clamp(-1, 1))
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)

def render_maps(json_path, mat_name, dir5, res, lr, epochs, tex_init):
    # epochs: list of 3 int. [0] is total epochs, [1] is epochs for latent, [2] is for noise, in each cycle.
    # tex_init: string. [], [PATH_TO_LATENT.pt], or [PATH_TO_LATENT.pt, PATH_TO_NOISE.pt]

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_path, dir5, device)
    svbrdf_obj.reference_dir = Path(json_path).parent
    # targets = svbrdf_obj.load_images_th(svbrdf_obj.target_dir, res)
    # svbrdf_obj.reference_dir = json_dir
    textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir / (mat_name + ".png"))

    render_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval(textures.to(device))
    svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)
    

def optim_ganlatent(json_path, mat_name, dir5, res, lr, epochs, tex_init, input_num):
    # epochs: list of 3 int. [0] is total epochs, [1] is epochs for latent, [2] is for noise, in each cycle.
    # tex_init: string. [], [PATH_TO_LATENT.pt], or [PATH_TO_LATENT.pt, PATH_TO_NOISE.pt]

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # device = th.device("cpu")

    svbrdf_obj = SvbrdfIO(json_path, dir5, device, input_num)
    svbrdf_obj.reference_dir = Path(json_path).parent
    # targets = svbrdf_obj.load_images_th(svbrdf_obj.target_dir, res)
    # svbrdf_obj.reference_dir = json_dir
    textures = svbrdf_obj.load_textures_th(svbrdf_obj.reference_dir / (mat_name + ".png"))

    render_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)
    rendereds = render_obj.eval(textures.to(device))

    renderer_obj = Microfacet(res, svbrdf_obj.n_of_imgs, svbrdf_obj.im_size, svbrdf_obj.cl, device)

    optim_obj = MaterialGANOptim(device, renderer_obj, ckp="ckp/materialgan.pth")
    optim_obj.load_targets(rendereds)
    
    # svbrdf_obj.save_images_th(rendereds, Path("./rendereds"))

    if tex_init == "auto":
        optim_obj.init_from(["ckp/latent_avg_W+_256.pt"])
        optim_obj.optim([80,10,10], lr, svbrdf_obj)
        loss_dif = optim_obj.loss_image
        optim_obj.init_from(["ckp/latent_const_W+_256.pt", "ckp/latent_const_N_256.pt"])
        optim_obj.optim([80,10,10], lr, svbrdf_obj)
        loss_spe = optim_obj.loss_image
        if loss_dif < loss_spe:
            print("Non-specular material!")
            optim_obj.init_from(["ckp/latent_avg_W+_256.pt"])
        else:
            print("Specular material!")
            optim_obj.init_from(["ckp/latent_const_W+_256.pt", "ckp/latent_const_N_256.pt"])
    else:
        optim_obj.init_from(tex_init)
    
    optim_obj.optim(epochs, lr, svbrdf_obj)

    svbrdf_obj.save_textures_th(optim_obj.textures, svbrdf_obj.optimize_dir, input_num)
    # rendereds = renderer_obj.eval(optim_obj.textures)
    # svbrdf_obj.save_images_th(rendereds, svbrdf_obj.rerender_dir)
