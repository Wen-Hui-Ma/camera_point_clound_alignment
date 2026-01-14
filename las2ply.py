import laspy
import numpy as np
import time
import json
import os
from plyfile import PlyData, PlyElement

def las_to_ply_fast(las_path, ply_path, save_offset=True):
    """
    将 LAS 文件转换为 PLY 格式，用于 3D Gaussian Splatting 初始化点云
    优化版本：减少精度损失
    
    Args:
        las_path: 输入的 LAS 文件路径
        ply_path: 输出的 PLY 文件路径（建议命名为 points3d.ply）
        save_offset: 是否保存偏移量到 JSON 文件
    """
    t0 = time.time()
    print(f"[1/5] 正在读取 LAS 文件: {las_path} ...")
    
    try:
        las = laspy.read(las_path)
    except Exception as e:
        print(f"错误: 无法读取 LAS 文件: {e}")
        return
    
    point_count = len(las.points)
    print(f"    -> 点云数量: {point_count:,}")
    
    # 检查 LAS 文件属性
    print(f"    -> LAS 文件信息:")
    print(f"       坐标范围: X[{las.x.min():.3f}, {las.x.max():.3f}], "
          f"Y[{las.y.min():.3f}, {las.y.max():.3f}], "
          f"Z[{las.z.min():.3f}, {las.z.max():.3f}]")
    if hasattr(las.header, 'scale'):
        print(f"       坐标缩放: {las.header.scale}")
    if hasattr(las.header, 'offset'):
        print(f"       坐标偏移: {las.header.offset}")

    # 优化点 1: 使用 float64 进行高精度计算，减少累积误差
    print(f"[2/5] 正在处理坐标（高精度计算）...")
    
    # 先用 float64 读取，保持最高精度
    points_f64 = np.zeros((point_count, 3), dtype=np.float64)
    points_f64[:, 0] = np.array(las.x, dtype=np.float64)
    points_f64[:, 1] = np.array(las.y, dtype=np.float64)
    points_f64[:, 2] = np.array(las.z, dtype=np.float64)
    
    # 在 float64 精度下计算重心
    center_offset = points_f64.mean(axis=0)
    print(f"    -> 全局偏移量 (Global Offset): [{center_offset[0]:.6f}, {center_offset[1]:.6f}, {center_offset[2]:.6f}]")
    
    # 中心化（在 float64 精度下）
    points_f64 -= center_offset
    
    # 最后转换为 float32（减少累积误差）
    points = points_f64.astype(np.float32)
    
    # 计算精度损失
    coord_range = points_f64.max(axis=0) - points_f64.min(axis=0)
    print(f"    -> 坐标范围: [{coord_range[0]:.3f}, {coord_range[1]:.3f}, {coord_range[2]:.3f}]")

    # 优化点 2: 颜色处理优化（使用感知均匀缩放）
    print(f"[3/5] 正在处理颜色（优化缩放）...")
    
    # 先将 ScaledArrayView 转换为 numpy 数组
    colors_orig = np.zeros((point_count, 3), dtype=np.uint16)
    colors_orig[:, 0] = np.array(las.red, dtype=np.uint16)
    colors_orig[:, 1] = np.array(las.green, dtype=np.uint16)
    colors_orig[:, 2] = np.array(las.blue, dtype=np.uint16)

    # 检查颜色范围
    max_color = colors_orig.max()
    min_color = colors_orig.min()
    print(f"    原始颜色范围: [{min_color}, {max_color}]")
    
    if max_color > 255:
        # 如果是 16 位颜色，使用优化的缩放方法
        if max_color > 65535:
            print(f"    警告: 检测到异常颜色值范围 (最大: {max_color})")
            colors_orig = np.clip(colors_orig, 0, 65535)
            max_color = 65535
        
        if max_color > 0:
            print(f"    检测到 16 位颜色，使用 Gamma 校正缩放（保留更多细节）...")
            # 方法1: Gamma 校正（2.2），更好地保留暗部细节
            colors_float = colors_orig.astype(np.float32) / max_color  # 归一化到 0-1
            colors_float = np.power(colors_float, 1.0/2.2)  # Gamma 校正
            colors = (colors_float * 255.0).astype(np.uint8)
        else:
            colors = colors_orig.astype(np.uint8)
    else:
        colors = colors_orig.astype(np.uint8)
    
    print(f"    最终颜色范围: [{colors.min()}, {colors.max()}]")
    if max_color > 255:
        print(f"    颜色压缩比: {max_color/255:.1f}:1")

    # 优化点 3: 检查是否有法向量信息
    print(f"[4/5] 正在处理法向量...")
    normals = np.zeros((point_count, 3), dtype=np.float32)
    
    # 检查 LAS 文件是否有法向量（某些 LAS 文件可能包含）
    has_normals = False
    if hasattr(las, 'normal_x') and hasattr(las, 'normal_y') and hasattr(las, 'normal_z'):
        try:
            normals[:, 0] = np.array(las.normal_x, dtype=np.float32)
            normals[:, 1] = np.array(las.normal_y, dtype=np.float32)
            normals[:, 2] = np.array(las.normal_z, dtype=np.float32)
            # 检查是否全为零
            if normals.max() > 0 or normals.min() < 0:
                has_normals = True
                print(f"    检测到法向量信息，已保留")
        except:
            pass
    
    if not has_normals:
        print(f"    未检测到法向量，使用全零（符合代码期望）")
    
    # 优化点 4: 使用 plyfile 保存，确保格式兼容
    print(f"[5/5] 正在保存 PLY: {ply_path} ...")
    
    # 定义 PLY 格式，与 dataset_readers.py 中的格式完全一致
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # 合并所有属性
    elements = np.empty(point_count, dtype=dtype)
    attributes = np.concatenate([points, normals, colors], axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # 创建并保存 PLY 文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element], text=False).write(ply_path)
    
    # 保存偏移量信息到 JSON 文件
    if save_offset:
        offset_file = ply_path.replace('.ply', '_offset.json')
        offset_info = {
            'center_offset': center_offset.tolist(),
            'original_range': {
                'x': [float(las.x.min()), float(las.x.max())],
                'y': [float(las.y.min()), float(las.y.max())],
                'z': [float(las.z.min()), float(las.z.max())]
            },
            'centered_range': {
                'x': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z': [float(points[:, 2].min()), float(points[:, 2].max())]
            },
            'color_info': {
                'original_max': int(max_color),
                'original_min': int(min_color),
                'compressed': max_color > 255
            }
        }
        with open(offset_file, 'w') as f:
            json.dump(offset_info, f, indent=2)
        print(f"    偏移量信息已保存到: {offset_file}")
    
    dt = time.time() - t0
    print(f"\n转换完成！耗时: {dt:.2f} 秒")
    print(f"输出文件: {ply_path}")
    print(f"注意: 点云已中心化，偏移量: [{center_offset[0]:.6f}, {center_offset[1]:.6f}, {center_offset[2]:.6f}]")

# --- 执行 ---
input_path = "House_data/colorized.las"
output_path = "House_data/points3d.ply"

if __name__ == "__main__":
    las_to_ply_fast(input_path, output_path)