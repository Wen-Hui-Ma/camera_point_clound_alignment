# check_camera_pointcloud_alignment.py
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData
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
        # 提取相机位置（C2W矩阵的平移部分）
        cam_pos = c2w[:3, 3]
        # 提取相机朝向（C2W矩阵的Z轴方向，相机看向的方向）
        cam_forward = c2w[:3, 2]
        cameras.append({
            'position': cam_pos,
            'forward': cam_forward,
            'c2w': c2w
        })
    return cameras, data

def project_points_to_camera(points, c2w, K, width, height):
    """将3D点投影到相机图像平面"""
    # 将点从世界坐标系转换到相机坐标系
    w2c = np.linalg.inv(c2w)
    
    # 齐次坐标
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # 转换到相机坐标系
    points_cam = (w2c @ points_homo.T).T[:, :3]
    
    # 过滤掉相机后面的点（z < 0）
    valid = points_cam[:, 2] > 0
    points_cam_valid = points_cam[valid]
    
    if len(points_cam_valid) == 0:
        return None, valid
    
    # 投影到图像平面
    points_2d = (K @ points_cam_valid.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    
    # 检查是否在图像范围内
    in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    
    return points_2d, valid

def check_alignment(ply_path, json_path, sample_cameras=10):
    """检测相机和点云是否对齐"""
    print("=" * 60)
    print("检测相机内外参与点云对齐")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载点云...")
    points = load_pointcloud(ply_path)
    print(f"   点云数量: {len(points)}")
    print(f"   点云范围:")
    print(f"     X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"     Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"     Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"   点云中心: [{points.mean(axis=0)[0]:.2f}, {points.mean(axis=0)[1]:.2f}, {points.mean(axis=0)[2]:.2f}]")
    
    print("\n2. 加载相机参数...")
    cameras, data = load_cameras(json_path)
    print(f"   相机数量: {len(cameras)}")
    
    # 提取相机位置
    cam_positions = np.array([cam['position'] for cam in cameras])
    print(f"   相机位置范围:")
    print(f"     X: [{cam_positions[:, 0].min():.2f}, {cam_positions[:, 0].max():.2f}]")
    print(f"     Y: [{cam_positions[:, 1].min():.2f}, {cam_positions[:, 1].max():.2f}]")
    print(f"     Z: [{cam_positions[:, 2].min():.2f}, {cam_positions[:, 2].max():.2f}]")
    print(f"   相机位置中心: [{cam_positions.mean(axis=0)[0]:.2f}, {cam_positions.mean(axis=0)[1]:.2f}, {cam_positions.mean(axis=0)[2]:.2f}]")
    
    # 2. 检查空间关系
    print("\n3. 检查空间关系...")
    point_center = points.mean(axis=0)
    cam_center = cam_positions.mean(axis=0)
    distance = np.linalg.norm(point_center - cam_center)
    print(f"   点云中心与相机中心距离: {distance:.2f}")
    
    if distance > 100:
        print(f"   ⚠️  警告: 距离较大，可能未对齐")
    else:
        print(f"   ✅ 距离合理，可能已对齐")
    
    # 3. 检查点云是否在相机视野内
    print("\n4. 检查点云投影到相机图像平面...")
    
    # 获取相机内参
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
    
    # 采样几个相机进行测试
    sample_indices = np.linspace(0, len(cameras)-1, min(sample_cameras, len(cameras)), dtype=int)
    
    in_view_ratios = []
    for idx in sample_indices:
        cam = cameras[idx]
        points_2d, valid = project_points_to_camera(points, cam['c2w'], K, width, height)
        
        if points_2d is not None:
            in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                      (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
            in_view_ratio = in_image.sum() / len(points)
            in_view_ratios.append(in_view_ratio)
            print(f"   相机 {idx}: {in_view_ratio*100:.1f}% 的点在图像范围内")
        else:
            print(f"   相机 {idx}: 没有点在相机前方")
            in_view_ratios.append(0)
    
    avg_ratio = np.mean(in_view_ratios)
    print(f"\n   平均: {avg_ratio*100:.1f}% 的点在图像范围内")
    
    if avg_ratio < 0.01:
        print(f"   ⚠️  警告: 几乎没有点在图像范围内，可能未对齐")
    elif avg_ratio < 0.1:
        print(f"   ⚠️  警告: 只有少量点在图像范围内，可能部分对齐")
    else:
        print(f"   ✅ 大部分点在图像范围内，对齐良好")
    
    # 4. 可视化（可选）
    print("\n5. 生成可视化...")
    try:
        fig = plt.figure(figsize=(15, 5))
        
        # 3D视图
        ax1 = fig.add_subplot(131, projection='3d')
        # 采样点云（太多点会卡）
        sample_points = points[::max(1, len(points)//10000)]
        ax1.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                   c='blue', s=0.1, alpha=0.3, label='Point Cloud')
        ax1.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2], 
                   c='red', s=20, label='Cameras')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D View: Point Cloud & Cameras')
        ax1.legend()
        
        # XY平面投影
        ax2 = fig.add_subplot(132)
        ax2.scatter(sample_points[:, 0], sample_points[:, 1], c='blue', s=0.1, alpha=0.3)
        ax2.scatter(cam_positions[:, 0], cam_positions[:, 1], c='red', s=20)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Projection')
        ax2.axis('equal')
        
        # XZ平面投影
        ax3 = fig.add_subplot(133)
        ax3.scatter(sample_points[:, 0], sample_points[:, 2], c='blue', s=0.1, alpha=0.3)
        ax3.scatter(cam_positions[:, 0], cam_positions[:, 2], c='red', s=20)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ Projection')
        ax3.axis('equal')
        
        plt.tight_layout()
        output_path = 'camera_pointcloud_alignment_check.png'
        plt.savefig(output_path, dpi=150)
        print(f"   可视化已保存到: {output_path}")
        plt.close()
    except Exception as e:
        print(f"   可视化失败: {e}")
    
    print("\n" + "=" * 60)
    print("检测完成")
    print("=" * 60)

# analyze_and_align.py
import json
import numpy as np
from plyfile import PlyData

def analyze_alignment(ply_path, json_path):
    """分析并建议对齐偏移量"""
    # 加载点云
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    point_center = points.mean(axis=0)
    
    # 加载相机
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cam_positions = []
    for frame in data['frames']:
        c2w = np.array(frame['transform_matrix'])
        cam_positions.append(c2w[:3, 3])
    
    cam_positions = np.array(cam_positions)
    cam_center = cam_positions.mean(axis=0)
    
    # 计算建议的偏移量
    suggested_offset = cam_center - point_center
    
    print("=" * 60)
    print("对齐分析")
    print("=" * 60)
    print(f"点云中心: {point_center}")
    print(f"相机中心: {cam_center}")
    print(f"当前偏移: {cam_center - point_center}")
    print(f"\n建议的对齐偏移量: {suggested_offset}")
    print(f"（将相机位置减去这个偏移量，使其与点云对齐）")
    
    return suggested_offset


# diagnose_camera_view.py
import json
import numpy as np
from plyfile import PlyData

def diagnose_camera_view(ply_path, json_path, camera_indices):
    """诊断特定相机为什么看不到点"""
    
    # 加载点云
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # 加载相机
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取相机内参
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
    
    for idx in camera_indices:
        frame = data['frames'][idx]
        c2w = np.array(frame['transform_matrix'])
        w2c = np.linalg.inv(c2w)
        
        # 相机位置
        cam_pos = c2w[:3, 3]
        # 相机朝向（Z轴方向）
        cam_forward = c2w[:3, 2]
        
        print(f"\n{'='*60}")
        print(f"相机 {idx} 诊断")
        print(f"{'='*60}")
        print(f"相机位置: {cam_pos}")
        print(f"相机朝向: {cam_forward}")
        
        # 转换到相机坐标系
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        points_cam = (w2c @ points_homo.T).T[:, :3]
        
        # 统计点在相机前方/后方的数量
        in_front = (points_cam[:, 2] > 0).sum()
        behind = (points_cam[:, 2] <= 0).sum()
        
        print(f"\n点在相机坐标系中的分布:")
        print(f"  前方 (z > 0): {in_front} ({in_front/len(points)*100:.1f}%)")
        print(f"  后方 (z <= 0): {behind} ({behind/len(points)*100:.1f}%)")
        
        if in_front == 0:
            print(f"  ⚠️  所有点都在相机后方！")
            # 计算点到相机的距离和方向
            points_to_cam = points - cam_pos
            distances = np.linalg.norm(points_to_cam, axis=1)
            directions = points_to_cam / (distances[:, None] + 1e-8)
            
            # 计算点方向与相机朝向的夹角
            cos_angles = np.dot(directions, cam_forward)
            angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi
            
            print(f"  最近的点距离: {distances.min():.2f}")
            print(f"  最远的点距离: {distances.max():.2f}")
            print(f"  点方向与相机朝向的平均夹角: {angles.mean():.1f}°")
            print(f"  点方向与相机朝向的最小夹角: {angles.min():.1f}°")
            continue
        
        # 投影前方点到图像平面
        points_cam_front = points_cam[points_cam[:, 2] > 0]
        points_2d = (K @ points_cam_front.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        # 统计投影结果
        in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                  (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        
        in_image_count = in_image.sum()
        out_image_count = len(points_2d) - in_image_count
        
        print(f"\n投影到图像平面:")
        print(f"  在图像范围内: {in_image_count} ({in_image_count/len(points)*100:.1f}%)")
        print(f"  在图像范围外: {out_image_count} ({out_image_count/len(points)*100:.1f}%)")
        
        if in_image_count == 0:
            print(f"  ⚠️  所有点都在图像范围外！")
            # 分析投影位置
            x_coords = points_2d[:, 0]
            y_coords = points_2d[:, 1]
            
            print(f"  投影X坐标范围: [{x_coords.min():.1f}, {x_coords.max():.1f}] (图像宽度: {width})")
            print(f"  投影Y坐标范围: [{y_coords.min():.1f}, {y_coords.max():.1f}] (图像高度: {height})")
            
            # 统计各个方向的超出情况
            left_out = (x_coords < 0).sum()
            right_out = (x_coords >= width).sum()
            top_out = (y_coords < 0).sum()
            bottom_out = (y_coords >= height).sum()
            
            print(f"  超出左侧: {left_out}")
            print(f"  超出右侧: {right_out}")
            print(f"  超出顶部: {top_out}")
            print(f"  超出底部: {bottom_out}")



if __name__ == "__main__":
    ply_path = "data/points3d.ply"
    json_path = "data/transforms_train.json"
    
    check_alignment(ply_path, json_path, sample_cameras=10)
    
    offset = analyze_alignment("data/points3d.ply", "data/transforms_train.json")
    print(f"\n可以在 align_camera_poses.py 中使用这个偏移量: {offset.tolist()}")
    
        # 诊断看不到点的相机
    diagnose_camera_view(
        "data/points3d.ply",
        "data/transforms_train.json",
        camera_indices=[1480, 1692]  # 看不到点的相机
    )