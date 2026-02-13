
import os
import torch
from torch.utils.data import DataLoader
from config import Config
from utils.dataset import IDRiD_Dataset
from utils.visualization import visualize_batch_rsm

def main():
    print("--- 1. 初始化配置 ---")
    Config.setup() # 确保目录存在
    print(f"IDRiD 数据集路径: {Config.IDRID_ROOT}")
    print(f"RSM Strides: {Config.RSM_STRIDES}")

    # --- 2. 初始化数据集 ---
    try:
        dataset = IDRiD_Dataset(
            img_dir=Config.TRAIN_IMG_DIR,
            mask_dir=Config.TRAIN_MASK_DIR,
            mask_suffix=Config.MASK_SUFFIX,
            img_size=Config.IMG_SIZE,
            rsm_strides=Config.RSM_STRIDES
        )
        print(f"✅ 数据集加载成功，共找到 {len(dataset)} 张图片。")
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        print("请检查 config.py 中的路径是否正确。")
        return

    # --- 3. 创建 DataLoader ---
    # 这里我们只取一个 Batch 来测试
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    print("--- 4. 读取一个 Batch 数据 ---")
    try:
        # 获取一个 Batch
        # Dataset 返回: image, mask, rsms
        images, original_masks, rsms = next(iter(dataloader))

        print("\n[数据维度检查]")
        print(f"Image Batch Shape: {images.shape} (B, C, H, W)")
        print(f"Original Mask Shape: {original_masks.shape} (B, 1, H, W)")

        print(f"\n[RSM 多尺度检查]")
        # rsms 是一个列表，对应 Config.RSM_STRIDES
        for i, rsm in enumerate(rsms):
            stride = Config.RSM_STRIDES[i]
            expected_size = Config.IMG_SIZE[0] // stride
            print(f"RSM Layer { i +1} (Stride {stride}):")
            print(f"  - Actual Shape: {rsm.shape}")
            print(f"  - Expected Size: {expected_size}x{expected_size}")

            # 检查数值范围 (RSM 应该是 0 或 1)
            unique_vals = torch.unique(rsm)
            print(f"  - Unique Values: {unique_vals.tolist()}")

            if len(unique_vals) > 2:
                print("  ⚠️ 警告: RSM 包含非 0/1 值，可能是插值错误！")

    except FileNotFoundError as e:
        print(f"❌ 读取文件失败: {e}")
        return
    except Exception as e:
        print(f"❌ 数据处理过程出错: {e}")
        return

    # --- 5. 可视化验证 ---
    print("\n--- 5. 生成可视化结果 ---")
    save_name = "data_check_preview.png"
    save_path = os.path.join(Config.VIS_DIR, save_name)

    try:
        visualize_batch_rsm(images, original_masks, rsms, save_path=save_path)
        print(f"✅ 可视化图片已保存至: {os.path.abspath(save_path)}")
        print("请打开图片，重点检查：")
        print("1. 原图和 Mask 的病灶位置是否对应？")
        print("2. RSM 的红色热力块是否大致覆盖了病灶区域？")
        print("3. 深层 RSM (Scale 1) 应该很粗糙，浅层 RSM (Scale 4) 应该较精细。")

    except Exception as e:
        print(f"❌ 可视化失败: {e}")

if __name__ == "__main__":
    main()
