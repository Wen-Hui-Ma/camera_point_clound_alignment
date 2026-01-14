# align_camera_poses_to_pointcloud.py
import json
import numpy as np
from pathlib import Path

# 从对齐分析得到的偏移量
point_cloud_offset = np.array([1.77808028, 3.05104896, 0.07509784])

# 文件路径
transforms_path = Path('data/transforms_train.json')
backup_path = Path('data/transforms_train.json.backup')

# 备份原文件
if not backup_path.exists():
    import shutil
    shutil.copy2(transforms_path, backup_path)
    print(f"已备份原文件到: {backup_path}")
else:
    print(f"备份文件已存在: {backup_path}")

# 加载数据
print("正在加载 transforms_train.json...")
with open(transforms_path, 'r') as f:
    data = json.load(f)

# 对齐所有相机位姿
print(f"正在对齐相机位姿（减去偏移量: {point_cloud_offset}）...")
aligned_count = 0

for i, frame in enumerate(data['frames']):
    c2w = np.array(frame['transform_matrix'])
    
    # 提取平移部分
    translation = c2w[:3, 3]
    
    # 减去点云偏移量
    new_translation = translation - point_cloud_offset
    
    # 更新位姿矩阵
    c2w[:3, 3] = new_translation
    frame['transform_matrix'] = c2w.tolist()
    
    aligned_count += 1
    if (i + 1) % 500 == 0:
        print(f"  已处理: {i + 1}/{len(data['frames'])}")

# 保存对齐后的文件
print(f"正在保存对齐后的 transforms_train.json...")
with open(transforms_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ 完成！已对齐 {aligned_count} 个相机位姿")
print(f"📁 原文件备份: {backup_path}")

# 验证对齐结果
print("\n验证对齐结果:")
positions = []
for frame in data['frames']:
    c2w = np.array(frame['transform_matrix'])
    pos = c2w[:3, 3]
    positions.append(pos)

positions = np.array(positions)
center = positions.mean(axis=0)
print(f"新的相机位置中心: {center}")
print(f"应该接近点云中心 [-0.03, -0.02, 0.00]")
print(f"\n建议运行 check_camera_pointcloud_alignment.py 验证对齐效果")