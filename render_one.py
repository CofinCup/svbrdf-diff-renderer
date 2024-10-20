import argparse
from imageio.v2 import imread, imwrite
from pathlib import Path
from src.scripts import render_envmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--material_path', type=str, default='/root/datasets/tests/')
    parser.add_argument('--result_path', type=str, default='/root/datasets/tests/rendereds')
    args = parser.parse_args()
    
    material_path = Path(args.material_path)
    result_path = Path(args.result_path)
    
    # 去掉result_path末尾的斜杠
    if result_path.as_posix().endswith('/'):
        print("[WARN] Remove trailing slash from result path")
        result_path = result_path.parent

    result_path.mkdir(parents=True, exist_ok=True)
    
    # 遍历material_path目录下的第一层.png文件，不递归子文件夹
    # png_files = [f for f in material_path.iterdir() if f.is_file() and f.suffix == ".png" ]
    # png_files = [f for f in material_path.iterdir() if f.is_file() and f.suffix == ".png" and f.name.startswith("multi_")]
    png_files = [f for f in material_path.iterdir() if f.is_file() and f.suffix == ".png" and ("aigc" in f.name)]
    
    if not png_files:
        print("[WARN] No .png files found in material directory:", material_path)
    
    for material_file in png_files:
        material_name = material_file.stem  # 获取文件名（不带扩展名）
        save_dir = result_path / material_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {material_file}")
        
        # 渲染环境贴图
        render_envmap(save_dir, material_file, 4096)
