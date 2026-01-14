#!/usr/bin/env python3
"""临时可视化脚本，运行后会被删除"""
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_pointcloud(ply_path):
    """加载点云"""
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return points

def load_cameras(json_path):
    """加载相机参数"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cameras = []
    for frame in data['frames']:
        c2w = np.array(frame['transform_matrix'])
        cam_pos = c2w[:3, 3]
        cam_forward = c2w[:3, 2]
        cameras.append({
            'position': cam_pos,
            'forward': cam_forward,
            'c2w': c2w
        })
    return cameras, data

def project_points_to_camera(points, c2w, K, width, height):
    """将3D点投影到相机图像平面"""
    w2c = np.linalg.inv(c2w)
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (w2c @ points_homo.T).T[:, :3]
    valid = points_cam[:, 2] > 0
    points_cam_valid = points_cam[valid]
    
    if len(points_cam_valid) == 0:
        return None, valid
    
    points_2d = (K @ points_cam_valid.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    
    return points_2d, valid

def main():
    ply_path = "data/points3d.ply"
    json_path = "data/transforms_train.json"
    
    print("=" * 60)
    print("相机与点云对齐检查")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载点云...")
    points = load_pointcloud(ply_path)
    print(f"   点云数量: {len(points)}")
    point_center = points.mean(axis=0)
    print(f"   点云中心: [{point_center[0]:.4f}, {point_center[1]:.4f}, {point_center[2]:.4f}]")
    print(f"   点云范围:")
    print(f"     X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"     Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"     Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    print("\n2. 加载相机参数...")
    cameras, data = load_cameras(json_path)
    print(f"   相机数量: {len(cameras)}")
    
    cam_positions = np.array([cam['position'] for cam in cameras])
    cam_center = cam_positions.mean(axis=0)
    print(f"   相机位置中心: [{cam_center[0]:.4f}, {cam_center[1]:.4f}, {cam_center[2]:.4f}]")
    print(f"   相机位置范围:")
    print(f"     X: [{cam_positions[:, 0].min():.2f}, {cam_positions[:, 0].max():.2f}]")
    print(f"     Y: [{cam_positions[:, 1].min():.2f}, {cam_positions[:, 1].max():.2f}]")
    print(f"     Z: [{cam_positions[:, 2].min():.2f}, {cam_positions[:, 2].max():.2f}]")
    
    # 2. 检查空间关系
    print("\n3. 对齐分析...")
    distance = np.linalg.norm(point_center - cam_center)
    offset = cam_center - point_center
    print(f"   点云中心与相机中心距离: {distance:.4f}")
    print(f"   当前偏移量: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
    
    if distance < 1.0:
        print(f"   ✅ 对齐良好（距离 < 1.0）")
    elif distance < 10.0:
        print(f"   ⚠️  距离较小但可能仍需微调")
    else:
        print(f"   ⚠️  距离较大，可能未对齐")
    
    # 3. 检查投影
    print("\n4. 检查点云投影...")
    fovx = data['camera_angle_x']
    width = data['w']
    height = data['h']
    fl_x = data.get('fl_x', width / (2 * np.tan(fovx / 2)))
    fl_y = data.get('fl_y', height / (2 * np.tan(fovx / 2)))
    cx = data.get('cx', width / 2)
    cy = data.get('cy', height / 2)
    
    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    sample_indices = np.linspace(0, len(cameras)-1, min(20, len(cameras)), dtype=int)
    in_view_ratios = []
    
    for idx in sample_indices:
        cam = cameras[idx]
        points_2d, valid = project_points_to_camera(points, cam['c2w'], K, width, height)
        
        if points_2d is not None:
            in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                      (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
            in_view_ratio = in_image.sum() / len(points)
            in_view_ratios.append(in_view_ratio)
        else:
            in_view_ratios.append(0)
    
    avg_ratio = np.mean(in_view_ratios)
    print(f"   平均 {avg_ratio*100:.2f}% 的点在图像范围内")
    
    # 4. 保存对齐数据
    alignment_data = {
        'point_cloud': {
            'center': point_center.tolist(),
            'range': {
                'x': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z': [float(points[:, 2].min()), float(points[:, 2].max())]
            },
            'num_points': len(points)
        },
        'cameras': {
            'center': cam_center.tolist(),
            'range': {
                'x': [float(cam_positions[:, 0].min()), float(cam_positions[:, 0].max())],
                'y': [float(cam_positions[:, 1].min()), float(cam_positions[:, 1].max())],
                'z': [float(cam_positions[:, 2].min()), float(cam_positions[:, 2].max())]
            },
            'num_cameras': len(cameras)
        },
        'alignment': {
            'distance': float(distance),
            'offset': offset.tolist(),
            'avg_in_view_ratio': float(avg_ratio)
        }
    }
    
    with open('alignment_data.json', 'w') as f:
        json.dump(alignment_data, f, indent=2)
    print(f"\n   对齐数据已保存到: alignment_data.json")
    
    # 5. 生成可视化
    print("\n5. 生成可视化...")
    fig = plt.figure(figsize=(18, 6))
    
    # 3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    sample_points = points[::max(1, len(points)//20000)]
    ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
               c='blue', s=0.1, alpha=0.3, label='Point Cloud')
    ax1.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
               c='red', s=20, label='Cameras', marker='^')
    ax1.scatter([point_center[0]], [point_center[1]], [point_center[2]], 
               c='green', s=100, marker='*', label='Point Cloud Center')
    ax1.scatter([cam_center[0]], [cam_center[1]], [cam_center[2]], 
               c='orange', s=100, marker='*', label='Camera Center')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: Point Cloud & Cameras')
    ax1.legend()
    
    # XY平面投影
    ax2 = fig.add_subplot(132)
    ax2.scatter(sample_points[:, 0], sample_points[:, 1], c='blue', s=0.1, alpha=0.3, label='Point Cloud')
    ax2.scatter(cam_positions[:, 0], cam_positions[:, 1], c='red', s=20, marker='^', label='Cameras')
    ax2.scatter([point_center[0]], [point_center[1]], c='green', s=100, marker='*', label='PC Center')
    ax2.scatter([cam_center[0]], [cam_center[1]], c='orange', s=100, marker='*', label='Cam Center')
    ax2.plot([point_center[0], cam_center[0]], [point_center[1], cam_center[1]], 
            'k--', linewidth=2, label=f'Distance: {distance:.2f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ平面投影
    ax3 = fig.add_subplot(133)
    ax3.scatter(sample_points[:, 0], sample_points[:, 2], c='blue', s=0.1, alpha=0.3, label='Point Cloud')
    ax3.scatter(cam_positions[:, 0], cam_positions[:, 2], c='red', s=20, marker='^', label='Cameras')
    ax3.scatter([point_center[0]], [point_center[2]], c='green', s=100, marker='*', label='PC Center')
    ax3.scatter([cam_center[0]], [cam_center[2]], c='orange', s=100, marker='*', label='Cam Center')
    ax3.plot([point_center[0], cam_center[0]], [point_center[2], cam_center[2]], 
            'k--', linewidth=2, label=f'Distance: {distance:.2f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    output_path = 'camera_pointcloud_alignment_visualization.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"   可视化已保存到: {output_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"对齐数据: alignment_data.json")
    print(f"可视化图像: {output_path}")

if __name__ == "__main__":
    main()
