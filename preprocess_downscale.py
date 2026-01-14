import argparse
import os
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_image(args):
    """处理单张图像缩放"""
    img_path, scale, output_dir, input_root = args
    
    try:
        with Image.open(img_path) as img:
            # 计算新尺寸
            new_width = int(img.width / scale)
            new_height = int(img.height / scale)
            
            # 使用 LANCZOS 滤波器进行缩放
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 保持相对目录结构
            rel_path = img_path.relative_to(input_root)
            save_path = output_dir / rel_path
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img_resized.save(save_path, quality=95)
            return True
    except Exception as e:
        # 仅在出错时打印，保持进度条整洁
        return f"Error processing {img_path}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Nerfstudio 图像多尺度预处理工具")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="原始高分辨率图像的根目录 (例如: data/images)")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8],
                        help="需要生成的缩放倍数列表 (默认: 2 4 8)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行进程数 (默认: CPU核心数)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="输出根目录 (默认: 与 input_dir 同级的 images_{scale})")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    # 确定进程数
    num_workers = args.num_workers if args.num_workers else cpu_count()
    
    tasks = []
    print(f"源目录: {input_dir}")
    print(f"缩放倍数: {args.scales}")

    for scale in args.scales:
        # 确定输出目录名称
        if args.output_root:
            output_dir = Path(args.output_root) / f"images_{scale}"
        else:
            # 默认逻辑: data/images -> data/images_2, data/images_4
            # 假设 input_dir 是 data/images，parent 是 data
            # 如果 input_dir 是 data/camera/left，这会在 data/camera/images_2 下生成，需注意
            # 为了稳健性，通常建议 input_dir 指向包含所有图片的顶层目录
            folder_name = f"{input_dir.name}_{scale}" # e.g., images_4
            output_dir = input_dir.parent / folder_name

        print(f" -> 准备生成: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 递归搜索图片
        files = list(input_dir.rglob("*"))
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 构建任务列表 (包含 input_dir 用于计算相对路径)
        for img_path in image_files:
            tasks.append((img_path, scale, output_dir, input_dir))

    if not tasks:
        print("未找到任何图像文件！")
        return

    print(f"开始处理 {len(tasks)} 个任务 (使用 {num_workers} 个进程)...")
    
    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 提高大量小任务时的性能
        results = list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks)))

    # 简单的错误统计
    errors = [r for r in results if r is not True]
    if errors:
        print(f"\n完成，但有 {len(errors)} 个错误:")
        for err in errors[:5]: # 只打印前5个错误
            print(err)
        if len(errors) > 5: print("...")
    else:
        print("\n所有图像处理成功！")

if __name__ == "__main__":
    main()