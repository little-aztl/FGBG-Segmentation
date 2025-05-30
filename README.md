# 一、环境准备

## 1. 配置代码运行环境

创建一个新的虚拟环境（Python 版本为 3.11.11），在虚拟环境中运行

```bash
pip install -r requirements.txt
```

## 2. 准备数据集

将 `weizmann2Images` 和 `weizmann2TruthOne` 两个文件夹放在代码库的根目录下。

# 二、运行脚本

测试代码性能：

```bash
python scripts/test.py
```

---

可视化初始聚类效果：

```bash
python scripts/visualize_cluster.py
```

初始聚类效果的可视化将会保存在 `sandbox/results/cluster_visualization` 中。

---

可视化分割效果：

```
python scripts/visualize_segmentation.py
```

分割效果的可视化将会保存在 `sandbox/results/segmentation_visualization` 中。
