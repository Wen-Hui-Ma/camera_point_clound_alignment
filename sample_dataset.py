import json
import random
import os
import shutil

# 配置路径
json_path = "data/transforms.json"
backup_path = "data/transforms_full_backup.json"
target_count = 100  # 你想要保留的图片数量

# 1. 安全备份（如果还没备份过）
if not os.path.exists(backup_path):
    print(f"正在备份原始文件到: {backup_path}")
    shutil.copy(json_path, backup_path)
else:
    print(f"检测到已有备份: {backup_path}，将基于该备份进行采样")
    # 如果已有备份，我们应该从备份读取，以防 transforms.json 已经被修改过
    json_path = backup_path

# 2. 读取数据
with open(json_path, 'r') as f:
    data = json.load(f)

total_frames = len(data['frames'])
print(f"原始帧数: {total_frames}")

if total_frames <= target_count:
    print("图片数量不足或等于 100，无需采样。")
    exit()

# 3. 随机采样
# 保持 frames 列表中的顺序，或者完全随机打乱
sampled_frames = random.sample(data['frames'], target_count)

# 更新数据
data['frames'] = sampled_frames

# 4. 覆盖写入 transforms.json
# Nerfstudio 默认只认这个文件名，所以我们覆盖它
output_path = "data/transforms.json"
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ 成功！已随机抽取 {len(data['frames'])} 帧。")
print(f"新文件已保存至: {output_path}")
print("现在运行训练命令，加载将瞬间完成。")