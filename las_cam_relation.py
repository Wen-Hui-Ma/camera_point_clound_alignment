# scripts/check_camera_las_relationship.py
"""
检测相机参数和LAS文件的关系
"""
import json
import numpy as np
import laspy
from pathlib import Path

def analyze_camera_las_relationship():
    """分析相机参数和LAS文件的关系"""
    
    print("=" * 60)
    print("相机参数和LAS文件关系检测")
    print("=" * 60)
    
    # 1. 读取LAS文件信息
    print("\n[1] 读取LAS文件信息...")
    las_path = "data/colorized.las"
    if not Path(las_path).exists():
        print(f"❌ LAS文件不存在: {las_path}")
        return
    
    las = laspy.read(las_path)
    print(f"✅ LAS文件点云数量: {len(las.points):,}")
    print(f"   X范围: [{las.x.min():.6f}, {las.x.max():.6f}]")
    print(f"   Y范围: [{las.y.min():.6f}, {las.y.max():.6f}]")
    print(f"   Z范围: [{las.z.min():.6f}, {las.z.max():.6f}]")
    
    # 计算LAS点云中心（全局偏移量）
    # laspy返回的是ScaledArrayView，需要转换为numpy数组
    las_x = np.array(las.x)
    las_y = np.array(las.y)
    las_z = np.array(las.z)
    las_center = np.array([las_x.mean(), las_y.mean(), las_z.mean()])
    print(f"   LAS点云中心（全局偏移量）: {las_center}")
    
    # 2. 读取相机参数
    print("\n[2] 读取相机参数...")
    transforms_path = "data/transforms_train.json"
    if not Path(transforms_path).exists():
        transforms_path = "data/transforms.json.backup"
    
    with open(transforms_path, 'r') as f:
        camera_data = json.load(f)
    
    print(f"✅ 相机数量: {len(camera_data['frames'])}")
    print(f"   相机模型: {camera_data.get('camera_model', '未知')}")
    print(f"   图像尺寸: {camera_data.get('w', '?')} x {camera_data.get('h', '?')}")
    print(f"   焦距: fx={camera_data.get('fl_x', '?')}, fy={camera_data.get('fl_y', '?')}")
    print(f"   主点: cx={camera_data.get('cx', '?')}, cy={camera_data.get('cy', '?')}")
    
    # 3. 分析相机位姿
    print("\n[3] 分析相机位姿...")
    camera_positions = []
    camera_timestamps = []
    
    for frame in camera_data['frames']:
        c2w = np.array(frame['transform_matrix'])
        pos = c2w[:3, 3]  # 相机位置（平移部分）
        camera_positions.append(pos)
        
        # 提取时间戳
        if 'timestamp' in frame:
            camera_timestamps.append(frame['timestamp'])
        elif 'file_path' in frame:
            # 从文件名提取时间戳
            filename = frame['file_path']
            try:
                timestamp = int(Path(filename).stem.split('_')[-1])
                camera_timestamps.append(timestamp)
            except:
                pass
    
    camera_positions = np.array(camera_positions)
    
    print(f"   相机位置范围:")
    print(f"     X: [{camera_positions[:, 0].min():.6f}, {camera_positions[:, 0].max():.6f}]")
    print(f"     Y: [{camera_positions[:, 1].min():.6f}, {camera_positions[:, 1].max():.6f}]")
    print(f"     Z: [{camera_positions[:, 2].min():.6f}, {camera_positions[:, 2].max():.6f}]")
    
    camera_center = camera_positions.mean(axis=0)
    print(f"   相机位置中心: {camera_center}")
    
    # 4. 检查对齐关系
    print("\n[4] 检查坐标系对齐关系...")
    
    # 从align_camera_poses.py获取偏移量
    point_cloud_offset = np.array([1.69544589, 2.88935979, 0.29927354])
    print(f"   点云偏移量（用于对齐）: {point_cloud_offset}")
    
    # 计算对齐后的相机中心
    aligned_camera_center = camera_center - point_cloud_offset
    print(f"   对齐后的相机中心: {aligned_camera_center}")
    print(f"   （应该接近 [0, 0, 0]，因为点云已中心化）")
    
    # 5. 检查时间戳关系
    print("\n[5] 检查时间戳关系...")
    if camera_timestamps:
        timestamps = np.array(camera_timestamps)
        print(f"   时间戳范围: [{timestamps.min()}, {timestamps.max()}]")
        print(f"   时间戳数量: {len(timestamps)}")
        print(f"   时间跨度: {(timestamps.max() - timestamps.min()) / 1e9:.2f} 秒")
        
        # 检查LAS文件是否有时间戳字段
        las_has_time = False
        if hasattr(las, 'gps_time') and len(las.gps_time) > 0:
            las_has_time = True
            print(f"   ✅ LAS文件包含GPS时间字段")
            print(f"      LAS时间范围: [{las.gps_time.min():.6f}, {las.gps_time.max():.6f}]")
        elif hasattr(las, 'timestamp'):
            las_has_time = True
            print(f"   ✅ LAS文件包含时间戳字段")
        else:
            print(f"   ⚠️  LAS文件不包含时间戳字段，无法进行时间对齐")
    else:
        print(f"   ⚠️  相机数据中没有时间戳信息")
    
    # 6. 空间关系总结
    print("\n[6] 空间关系总结:")
    print(f"   LAS点云中心: {las_center}")
    print(f"   相机位置中心（原始）: {camera_center}")
    print(f"   相机位置中心（对齐后）: {aligned_camera_center}")
    print(f"   偏移量差异: {np.abs(camera_center - las_center - point_cloud_offset).max():.6f}")
    print(f"   （如果差异很小，说明对齐正确）")
    
    # 7. 坐标系一致性检查
    print("\n[7] 坐标系一致性检查:")
    print(f"   相机模型: {camera_data.get('camera_model', '未知')}")
    if camera_data.get('camera_model') == 'OPENCV':
        print(f"   ✅ 使用OpenCV/COLMAP坐标系（Y down, Z forward）")
    else:
        print(f"   ⚠️  相机模型不是OPENCV，可能需要坐标转换")
    
    print("\n" + "=" * 60)
    print("检测完成！")
    print("=" * 60)

if __name__ == "__main__":
    analyze_camera_las_relationship()