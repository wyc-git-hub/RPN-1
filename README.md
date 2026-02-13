RPN_Phase1_Visual

项目结构已根据 path.txt 生成最小可运行框架。

快速开始

1. 安装依赖：
   pip install -r requirements.txt

2. 修改 `config.py` 中的 `IDRID_ROOT` 路径指向本地 IDRiD 数据集。

3. 运行数据检查：
   python check_data.py

说明
- `utils/dataset.py` 中使用了占位的 RSM 生成（对 GT 做模糊），用于调试和可视化。
- 若要训练模型，在 `train_peripheral.py` 中补充损失函数与训练目标。

# RPN-PVB
