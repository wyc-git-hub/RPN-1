import os
import torch


class Config:
    # =========================================================================
    # 1. 路径配置 (Path Configuration)
    # =========================================================================

    # [重要] 请修改为你本地 IDRiD 数据集的根目录路径
    # 官方解压后通常包含 "A. Segmentation", "B. Disease Grading" 等文件夹
    IDRID_ROOT = r'C:\Users\wyc\OneDrive\桌面\RPN_PVB\data\IDRiD'

    # --- 训练集路径 (A. Segmentation) ---
    # 原始图像路径
    TRAIN_IMG_DIR = os.path.join(
        IDRID_ROOT,
        'A. Segmentation', '1. Original Images', 'a. Training Set'
    )

    # Ground Truth 路径
    # IDRiD 将不同病灶分开存放：
    # 1. Microaneurysms (微动脉瘤) -> 文件后缀 usually _MA.tif
    # 2. Haemorrhages (出血点) -> _HE.tif
    # 3. Hard Exudates (硬性渗出) -> _EX.tif
    # 4. Soft Exudates (软性渗出) -> _SE.tif
    # 论文中 RPN 主要针对难样本 (如 MA)，这里默认配置为微动脉瘤 (MA)
    TRAIN_MASK_DIR = os.path.join(
        IDRID_ROOT,
        'A. Segmentation', '2. All Segmentation Groundtruths', 'a. Training Set', '2. Haemorrhages'
    )

    # 对应的 Mask 文件后缀 (用于 Dataset 类中自动匹配文件名)
    MASK_SUFFIX = '_HE.tif'

    # --- 结果保存路径 ---
    # 训练权重保存目录

    CHECKPOINT_DIR = './checkpoints'
    # 可视化结果保存目录
    VIS_DIR = './visualization_results'

    # =========================================================================
    # 2. 数据处理参数 (Data Preprocessing)
    # =========================================================================

    # 论文 Section 4.3.1: "All IDRiD images were scaled to 640 × 640."
    IMG_SIZE = (640, 640)

    # 输入通道数 (RGB=3)
    IN_CHANNELS = 3

    # 类别数 (对于单病灶分割通常是 1，即前景/背景)
    NUM_CLASSES = 1

    # =========================================================================
    # 3. RPN 模型参数 (Model Parameters)
    # =========================================================================

    # [核心] 周边视觉分支的下采样倍率 (Strides)
    # 对应 U-Net 解码器的 4 个层级。
    # 假设 Input=640x640:
    # Stride 32 -> Mask Size 20x20   (Deepest Layer)
    # Stride 16 -> Mask Size 40x40
    # Stride 8  -> Mask Size 80x80
    # Stride 4  -> Mask Size 160x160 (Shallowest Layer)
    RSM_STRIDES = [8, 4, 2, 1]

    # 论文公式 (11) 中的 lambda 权重，用于平衡不同尺度的 Loss
    # 论文中未明确给出具体数值，通常深层(语义强)权重低，浅层(细节多)权重高，或者设为 1.0
    LAMBDAS = [1.0, 1.0, 1.0, 1.0]

    # =========================================================================
    # 4. 训练超参数 (Training Hyperparameters)
    # =========================================================================

    BATCH_SIZE = 4  # 根据你的显存调整 (建议 4 或 8)
    LEARNING_RATE = 1e-4  # 常用初始学习率
    NUM_EPOCHS = 500  # 总训练轮数

    # 硬件设置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4  # DataLoader 线程数 (Windows下建议设为0调试，Linux设为4-8)
    OUTPUT_DIR = './output'  # 输出目录
    # 随机种子 (保证复现性)
    SEED = 42

    # 设置PVB和CVB分支
    PVB_LIST = ['OUT_1', 'OUT_2', 'OUT_3', 'OUT_4']
    CVB_LIST = ['OUT_1', 'OUT_2', 'OUT_3', 'OUT_4']
    # =========================================================================
    # 5. 自动创建目录
    # =========================================================================
    @classmethod
    def setup(cls):
        """自动创建必要的文件夹"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.VIS_DIR, exist_ok=True)


# 运行一次 setup 以确保文件夹存在
Config.setup()