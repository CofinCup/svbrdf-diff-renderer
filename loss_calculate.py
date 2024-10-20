import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from src.scripts import render_for_loss

def calculate_mse(imageA, imageB):
    imageA = imageA.astype("float")/255
    imageB = imageB.astype("float")/255
    
    mse = np.mean((imageA - imageB) ** 2)
    return mse

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
    input_numbers = range(1, 11)  # input_number范围
    mse_values = {"random": {}, "uniform": {}, "focus": {}}  # 存储每个 test_type 下 input_number 的 mse 总和及计数

    for test_type in ["random", "uniform", "focus"]:
        nn_render_folder = "/root/test_data/rendereds/"
        nn_output_folder = f"/root/test_data/materialgan/nonspecular/{dataset}/{test_type}"

        for mat_name in tqdm(sorted(os.listdir(nn_output_folder))):
            mat_type = mat_name.rsplit('_', 1)[0]
            gt_render_folder = "/root/tmp/loss_render/gt"
            gt_mat_name = f"{mat_name}/{mat_name}"
            gt_folder = f"/root/datasets/montage/{dataset}/{mat_type}"
            rerender(gt_folder, gt_mat_name, gt_render_folder)

            for input_pic_number in range(1, 5 if test_type == "focus" else 11):
                png_file_name = f"{input_pic_number}_tex"
                if not os.path.exists(f"/root/test_data/materialgan/nonspecular/{dataset}/{test_type}/" + mat_name + "/" + png_file_name + ".png"):
                    print(f"/root/test_data/materialgan/nonspecular/{dataset}/{test_type}/" + mat_name + "/" + png_file_name + ".png")
                    continue
                rerender(nn_output_folder, mat_name + "/" + png_file_name, nn_render_folder)
                
                mse = calculate_by_folder(nn_render_folder + "/" + mat_name + "/" + png_file_name,
                                          gt_render_folder + "/" + gt_mat_name)
                
                if input_pic_number not in mse_values[test_type]:
                    mse_values[test_type][input_pic_number] = {"total_mse": 0, "count": 0}
                
                mse_values[test_type][input_pic_number]["total_mse"] += mse
                mse_values[test_type][input_pic_number]["count"] += 1
                
                print(f"{mat_name} {png_file_name}: {mse}")

    # 计算平均 MSE 并画图
    markers = {"random": "o", "uniform": "s", "focus": "^"}  # o: 圆, s: 方块, ^: 三角形
    plt.figure()

    for test_type in ["random", "uniform", "focus"]:
        avg_mse = []
        for input_pic_number in input_numbers:
            if input_pic_number in mse_values[test_type]:
                total_mse = mse_values[test_type][input_pic_number]["total_mse"]
                count = mse_values[test_type][input_pic_number]["count"]
                avg_mse.append(total_mse / count)
            else:
                avg_mse.append(None)  # 对于 focus 类型的多余 input_number，不绘制点

        plt.plot(input_numbers, avg_mse, label=test_type, marker=markers[test_type])

    plt.xlabel('Input Number')
    plt.ylabel('Average MSE')
    plt.title(f'Average MSE vs Input Number for {dataset}')
    plt.legend()
    plt.savefig(f"/root/test_data/materialgan/nonspecular/{dataset}/mse_vs_input_number.png")