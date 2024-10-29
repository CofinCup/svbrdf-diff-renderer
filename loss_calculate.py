import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from src.scripts import render_for_loss
from src.imageio import imread, imwrite, img9to1, tex4to1

def calculate_mse(imageA, imageB):
    imageA = imageA.astype("float")/255
    imageB = imageB.astype("float")/255
    
    mse = np.mean((imageA - imageB) ** 2)
    return mse

def calculate_mse_for01(imageA, imageB):
    imageA = imageA.astype("float")
    imageB = imageB.astype("float")
    
    mse = np.mean((imageA - imageB) ** 2)
    return mse

def calculate_map_mse(gt_image, nn_image):
    """
    计算单张ground truth贴图与neural network输出贴图的MSE
    """
    print(gt_image.shape, nn_image.shape)
    gt_image_resized = cv2.resize(gt_image, (256, 256))  # 调整gt图片大小为256x256
    nn_image_resized = cv2.resize(nn_image, (256, 256))  # nn输出的图片也调整为256x256

    return calculate_mse_for01(gt_image_resized, nn_image_resized)

def calculate_map_total_mse(gt_image, nn_image):
    print(gt_image.shape, nn_image.shape)
    gt_image_resized = cv2.resize(gt_image, (1024, 256))  # 调整gt图片大小为256x256
    nn_image_resized = cv2.resize(nn_image, (1024, 256))  # nn输出的图片也调整为256x256

    return calculate_mse_for01(gt_image_resized, nn_image_resized)

def calculate_texture_maps_mse(gt_texture_path, nn_texture_path):
    """
    计算四张子贴图的MSE：法线、漫反射、粗糙度、镜面反射
    """
    # 加载gt与nn的svbrdf贴图
    print(gt_texture_path)
    gt_svbrdf = imread(gt_texture_path, "srgb")
    nn_svbrdf = imread(nn_texture_path, "srgb")
    
    
    if gt_svbrdf.shape[1] != 4 * gt_svbrdf.shape[0] or nn_svbrdf.shape[1] != 4 * nn_svbrdf.shape[0]:
        print(f"贴图宽度: {gt_svbrdf.shape[1]}, {nn_svbrdf.shape[1]}, 高度: {gt_svbrdf.shape[0]}, {nn_svbrdf.shape[0]}")
        raise ValueError("贴图宽度不符合4倍高度的要求")
    
    # 分割gt和nn的四张贴图
    height = gt_svbrdf.shape[0]
    gt_normal = gt_svbrdf[:, :height]
    gt_diffuse = gt_svbrdf[:, height:2*height]
    gt_roughness = gt_svbrdf[:, 2*height:3*height]
    gt_specular = gt_svbrdf[:, 3*height:]

    height = nn_svbrdf.shape[0]
    nn_normal = nn_svbrdf[:, :height]
    nn_diffuse = nn_svbrdf[:, height:2*height]
    nn_roughness = nn_svbrdf[:, 2*height:3*height]
    nn_specular = nn_svbrdf[:, 3*height:]

    # 分别计算每张子贴图的MSE
    map_mse = calculate_map_total_mse(gt_svbrdf,nn_svbrdf)
    normal_mse = calculate_map_mse(gt_normal, nn_normal)
    diffuse_mse = calculate_map_mse(gt_diffuse, nn_diffuse)
    roughness_mse = calculate_map_mse(gt_roughness, nn_roughness)
    specular_mse = calculate_map_mse(gt_specular, nn_specular)

    # 将所有子贴图的MSE汇总并返回
    mse_dict = {
        "map_mse" : map_mse,
        "normal_mse": normal_mse,
        "diffuse_mse": diffuse_mse,
        "roughness_mse": roughness_mse,
        "specular_mse": specular_mse
    }
    return mse_dict

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):  # 按顺序读取文件
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def calculate_by_folder(result_folder, ground_truth_folder):
    # 加载 result 和 ground truth 图片
    result_images = load_images_from_folder(result_folder)
    ground_truth_images = load_images_from_folder(ground_truth_folder)
    
    if len(result_images) != len(ground_truth_images):
        print("图片数量不一致")
        return
    
    total_mse = 0
    for i in range(len(result_images)):
        mse = calculate_mse(result_images[i], ground_truth_images[i])
        total_mse += mse

    avg_mse = total_mse / len(result_images)
    return avg_mse

def rerender(mat_folder, mat_name, result_folder):
    # 渲染材质，并将结果保存到 result_folder
    render_for_loss(Path(mat_folder) / (mat_name + ".png"), ["", "target", "", "", result_folder + "/" + mat_name])
    
if __name__ == "__main__":
    dataset = "multi-pics-with-noise_v2"
    input_numbers = range(1, 11)
    mse_values = {"random": {}, "uniform": {}}
    mse_types = ["map", "normal", "diffuse", "roughness", "specular", "render"]

    # 初始化 MSE 数据字典
    for test_type in ["random", "uniform", "focus"]:
        for mse_type in mse_types:
            mse_values[test_type][mse_type] = {"total_mse": {}, "count": {}}
    
    nn_render_folder = "/root/test_data/rendereds/"

    for test_type in ["random", "uniform", "focus"]:
        nn_output_folder = f"/root/test_data/alltest_svbrdf/{test_type}"

        for mat_name in tqdm(sorted(os.listdir(nn_output_folder))):
            mat_type = mat_name.rsplit('_', 1)[0]
            gt_render_folder = "/root/tmp/loss_render/gt"
            gt_mat_name = f"{mat_name}/{mat_name}"
            gt_folder = f"/root/datasets/montage/{dataset}/{mat_type}"
            rerender(gt_folder, gt_mat_name, gt_render_folder)

            for input_pic_number in range(1, 5 if test_type == "focus" else 10):
                png_file_name = f"{input_pic_number}_tex"
                rerender(nn_output_folder, mat_name + "/" + png_file_name, nn_render_folder)

                # 计算材质贴图的 MSE
                mse_dict = calculate_texture_maps_mse(
                    gt_folder + "/" + gt_mat_name + ".png",
                    nn_output_folder + "/" + mat_name + "/" + png_file_name + ".png"
                )

                # 计算渲染的 MSE
                render_mse = calculate_by_folder(
                    gt_render_folder + "/" + gt_mat_name,
                    nn_render_folder + "/" + mat_name + "/" + png_file_name
                )

                # 累加各个子贴图的 MSE
                for mse_type in mse_types[:-1]:  # 处理 normal, diffuse, roughness, specular 的 MSE
                    mse_type_key = f"{mse_type}_mse"
                    if input_pic_number not in mse_values[test_type][mse_type]["total_mse"]:
                        mse_values[test_type][mse_type]["total_mse"][input_pic_number] = 0
                        mse_values[test_type][mse_type]["count"][input_pic_number] = 0
                    mse_values[test_type][mse_type]["total_mse"][input_pic_number] += mse_dict[mse_type_key]
                    mse_values[test_type][mse_type]["count"][input_pic_number] += 1

                # 累加 render 的 MSE
                if input_pic_number not in mse_values[test_type]["render"]["total_mse"]:
                    mse_values[test_type]["render"]["total_mse"][input_pic_number] = 0
                    mse_values[test_type]["render"]["count"][input_pic_number] = 0
                mse_values[test_type]["render"]["total_mse"][input_pic_number] += render_mse
                mse_values[test_type]["render"]["count"][input_pic_number] += 1

                print(f"{mat_name} {png_file_name} MSE: {mse_dict}, render MSE: {render_mse}")

    # 计算平均 MSE 并画图
    markers = {"random": "o", "uniform": "s", "focus": "^"}  # o: 圆, s: 方块, ^: 三角形
    for mse_type in mse_types:
        plt.figure()
        for test_type in ["random", "uniform"]:
            avg_mse = []
            for input_pic_number in input_numbers:
                if input_pic_number in mse_values[test_type][mse_type]["total_mse"]:
                    total_mse = mse_values[test_type][mse_type]["total_mse"][input_pic_number]
                    count = mse_values[test_type][mse_type]["count"][input_pic_number]
                    avg_mse.append(total_mse / count)
                else:
                    avg_mse.append(None)  # 对于 focus 类型的多余 input_number，不绘制点

            plt.plot(input_numbers, avg_mse, label=test_type, marker=markers[test_type])

        plt.xlabel('Input Number')
        plt.ylabel(f'Average {mse_type.capitalize()} MSE')
        plt.title(f'Average {mse_type.capitalize()} MSE vs Input Number for aigc')
        plt.legend()
        plt.savefig(f"/root/test_data/aigc/{test_type}/{mse_type}_mse.png")
        print(f"图片保存到 /root/test_data/aigc/{test_type}/{mse_type}_mse.png")
