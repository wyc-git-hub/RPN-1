import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import time

# --- 导入自定义模块 ---
from config import Config
from models.rpn_phase1 import RPN_Phase1
from utils.dataset import IDRiD_Dataset
from utils.loss import PeripheralLoss
from utils.visualization import visualize_batch_rsm


# 可选: 强制同步 CUDA 报错 (调试时开启，平时注释掉以提高速度)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    """训练一个 Epoch"""
    model.train()
    running_loss = 0.0

    # 使用 tqdm 显示进度条
    loop = tqdm(loader, desc=f"Train Epoch {epoch_idx}/{Config.NUM_EPOCHS}")

    # [修改点 1] 正确解包: dataset 现在返回 image, mask(RGB), rsm_gts
    for batch_idx, (images, mask, rsm_gts, pfm_gt) in enumerate(loop):
        # 1. 数据搬运到 GPU
        images = images.to(device)

        # rsm_gts 是列表，需要列表推导式搬运
        # [注意] Loss 内部会自动处理 clamp 和维度对齐，这里只需搬运
        rsm_gts = [r.to(device) for r in rsm_gts]
        pfm_gt.to(device)
        # 2. 前向传播
        # outputs: [pred_rsm1, pred_rsm2, pred_rsm3, pred_rsm4]
        outputs, pfm_logits = model(images)

        # [修改点 2] 移除手动裁剪代码
        # 之前的 manual cropping 代码已删除，因为 utils/loss.py 现在会自动处理
        # 640 vs 641 的一像素差异。

        # 3. 可视化 (仅在每个 Epoch 的第一个 Batch)
        if batch_idx == 0:
            # 构造保存路径
            vis_save_path = os.path.join(
                Config.OUTPUT_DIR,
                f"epoch{epoch_idx}_batch{batch_idx}_rsm{Config.MASK_SUFFIX}.png"
            )

            # [修改点 3] 传入 mask (即使在 GPU 上也没关系，可视化函数会处理)
            # mask 不需要 .to(device) 除非后续计算需要，但传给可视化函数无所谓
            visualize_batch_rsm(
                images,
                mask,  # 传入原始 RGB Mask
                rsm_gts,  # GT 列表
                outputs,  # 预测列表
                save_path=vis_save_path
            )

        # 4. 计算 Loss
        loss, loss_dict = criterion(outputs, rsm_gts, pfm_logits, pfm_gt)

        # 5. 反向传播与优化
        optimizer.zero_grad()  # 清空过往梯度
        loss.backward()  # 计算当前梯度
        optimizer.step()  # 根据梯度更新参数

        # 6. 更新统计
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """验证阶段 (不计算梯度)"""
    model.eval()
    running_loss = 0.0
    scale_losses = [0.0] * 4

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating")
        # 验证阶段我们不需要 mask，用 _ 忽略即可
        for images, _, rsm_gts in loop:
            images = images.to(device)
            rsm_gts = [r.to(device) for r in rsm_gts]

            outputs = model(images)

            # Loss 计算 (自动处理尺寸对齐)
            loss, loss_dict = criterion(outputs, rsm_gts)

            running_loss += loss.item()

            # 累加各尺度的 Loss
            for i in range(4):
                scale_losses[i] += loss_dict.get(f"loss_scale_{i + 1}", 0)

    avg_loss = running_loss / len(loader)
    avg_scale_losses = [l / len(loader) for l in scale_losses]

    return avg_loss, avg_scale_losses


def main():
    # 1. 硬件配置
    print(f"Using device: {Config.DEVICE}")
    Config.setup()

    # 2. 准备数据集
    full_dataset = IDRiD_Dataset(
        img_dir=Config.TRAIN_IMG_DIR,
        mask_dir=Config.TRAIN_MASK_DIR,
        mask_suffix=Config.MASK_SUFFIX,
        img_size=Config.IMG_SIZE,
        rsm_strides=Config.RSM_STRIDES
    )

    # 划分数据集 (90% Train, 10% Val)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子，保证可复现
    )

    print(f"Total images: {len(full_dataset)}")
    print(f"Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # 建议开启，防止最后一个 Batch 大小不足导致某些计算报错
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # 3. 初始化模型
    model = RPN_Phase1(
        n_channels=Config.IN_CHANNELS,
        n_classes=Config.NUM_CLASSES
    ).to(Config.DEVICE)

    # 4. 定义损失函数和优化器
    criterion = PeripheralLoss(lambdas=Config.LAMBDAS)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # 5. 训练循环
    best_val_loss = float('inf')
    print("--- Start Training Phase 1: Peripheral Vision Branch ---")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # --- Train ---
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, epoch + 1
        )

        # --- Validation ---
        val_loss, val_scale_losses = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        # --- Scheduler Step ---
        scheduler.step(val_loss)

        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        # --- 打印日志 ---
        print(f"\nEpoch [{epoch + 1}/{Config.NUM_EPOCHS}] Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_mins:.1f} min")
        print(f"Scale Losses (Deep -> Shallow): "
              f"L1={val_scale_losses[0]:.4f}, L2={val_scale_losses[1]:.4f}, "
              f"L3={val_scale_losses[2]:.4f}, L4={val_scale_losses[3]:.4f}")

        # --- 保存最佳模型 ---
        if val_loss < best_val_loss:
            print(f"✅ Validation Loss Improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            save_path = os.path.join(Config.CHECKPOINT_DIR, 'best_rpn_phase1.pth')
            torch.save(model.state_dict(), save_path)

        print("-" * 60)

    print("Training Complete. Best model saved at:", os.path.join(Config.CHECKPOINT_DIR, 'best_rpn_phase1.pth'))


if __name__ == "__main__":
    main()