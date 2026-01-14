import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_single_image(args):
    """处理单张图像的去畸变"""
    frame, input_image_dir, output_image_dir, undistort_intrinsic, undistort_size = args
    
    try:
        # 读取原始内参和畸变参数
        K = np.array([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        D = np.array([frame['k1'], frame['k2'], frame['k3'], frame['k4']], dtype=np.float64)
        
        # 读取原始图像
        input_image_path = os.path.join(input_image_dir, frame['file_path'])
        if not os.path.exists(input_image_path):
            return None, f"文件不存在: {input_image_path}"
        
        img = cv2.imread(input_image_path)
        if img is None:
            return None, f"无法读取图像: {input_image_path}"
        
        # 去畸变
        new_K = undistort_intrinsic.astype(np.float64)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K,
            undistort_size,
            cv2.CV_16SC2
        )
        undistorted_img = cv2.remap(img, map1, map2, 
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
        
        # 保存去畸变图像
        output_image_path = os.path.join(output_image_dir, frame['file_path'])
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, undistorted_img)
        
        return True, None
    except Exception as e:
        return None, str(e)

def undistort_fisheye_images(
    transforms_path,
    output_transforms_path,
    output_image_dir,
    input_image_dir,
    num_workers=None
):
    """
    将鱼眼图像去畸变为透视投影图像（多进程版本）
    """
    # 读取原始transforms
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # 获取去畸变相机参数
    if 'undistort_camera_model' not in data:
        raise ValueError("transforms.json中缺少'undistort_camera_model'字段")
    
    undistort_model = data['undistort_camera_model']
    undistort_intrinsic = np.array(undistort_model['intrinsic'], dtype=np.float64)
    
    # 验证内参矩阵格式
    if undistort_intrinsic.shape != (3, 3):
        raise ValueError(f"undistort_intrinsic必须是3x3矩阵，当前形状: {undistort_intrinsic.shape}")
    
    undistort_width = undistort_model['width']
    undistort_height = undistort_model['height']
    undistort_size = (undistort_width, undistort_height)
    
    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    
    # 准备新的transforms数据
    new_data = {
        "camera_angle_x": 2 * np.arctan(undistort_width / (2 * undistort_intrinsic[0, 0])),
        "camera_angle_y": 2 * np.arctan(undistort_height / (2 * undistort_intrinsic[1, 1])),
        "fl_x": float(undistort_intrinsic[0, 0]),
        "fl_y": float(undistort_intrinsic[1, 1]),
        "cx": float(undistort_intrinsic[0, 2]),
        "cy": float(undistort_intrinsic[1, 2]),
        "w": undistort_width,
        "h": undistort_height,
        "frames": []
    }
    
    # 准备处理任务
    frames = data['frames']
    if num_workers is None:
        num_workers = min(cpu_count(), 16)
    
    # 准备参数列表
    tasks = [(frame, input_image_dir, output_image_dir, undistort_intrinsic, undistort_size) 
             for frame in frames]
    
    # 多进程处理
    success_count = 0
    error_count = 0
    
    print(f"使用 {num_workers} 个进程并行处理 {len(tasks)} 张图像...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, tasks),
            total=len(tasks),
            desc="去畸变图像"
        ))
    
    # 处理结果并构建新的transforms
    for frame, result in zip(frames, results):
        success, error = result
        if success:
            # 添加到新transforms
            new_frame = {
                "transform_matrix": frame['transform_matrix'],
                "file_path": frame['file_path']  # 相对路径保持不变
            }
            new_data['frames'].append(new_frame)
            success_count += 1
        else:
            error_count += 1
            if error:
                print(f"警告: {error}")
    
    # 保存新的transforms
    with open(output_transforms_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"\n完成！")
    print(f"  成功处理: {success_count} 张")
    print(f"  失败: {error_count} 张")
    print(f"去畸变图像保存在: {output_image_dir}")
    print(f"新的transforms保存在: {output_transforms_path}")

if __name__ == "__main__":
    undistort_fisheye_images(
        transforms_path="data/transforms.json",
        output_transforms_path="data/transforms_pinhole.json",
        output_image_dir="data/images_pinhole",
        input_image_dir="data/camera",  # 图像在data/camera/left和data/camera/right目录下
        num_workers=None  # 自动选择
    )