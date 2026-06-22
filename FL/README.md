# Federated Flow Detection

一个面向 CICIDS2017 流量特征的联邦学习入侵检测项目。项目先将原始流量 CSV 预处理成可联邦划分的训练数据，再通过 FedAvg 训练全局模型，最后支持对样本做多分类预测与“异常/正常”二值判定。

## 项目目标

- 将 CICIDS2017 或相似的 flow-feature CSV 统一清洗、标准化并切分为 10 个联邦客户端数据集
- 使用联邦学习训练全局多分类模型
- 同时支持两类模型结构：
  - `CNN` 版本：把标准化后的特征重排为二维灰度图输入
  - `MLP` 版本：直接输入一维特征向量
- 对推理结果额外给出异常分数，默认以 `非 BENIGN = 异常` 的方式进行二值判定
- 提供对真实 CSV、验证 NPZ、联邦客户端 NPZ 的推理与评估脚本

## 目录结构

```text
fl/
├─ federated/
│  ├─ train_fl.py              # 联邦训练（CNN 版本）
│  ├─ train_fl_mlp.py          # 联邦训练（MLP 版本）
│  ├─ inference.py             # CNN 全局模型推理与评估
│  ├─ inference_mlp.py         # MLP 全局模型推理与评估
│  ├─ inference_real_csv.py    # 对真实 CSV 做逐行预测（无标签也可用）
│  └─ __init__.py
├─ scripts/
│  ├─ prepare_cicids2017.py    # 预处理 CICIDS2017 并生成联邦 NPZ
│  ├─ prepare_cicids2017_train_split.py
│  ├─ pcap_to_simple_flow_csv.py
│  └─ csv_to_test_npz.py       # 将 CSV 转换为可评估的 test.npz
├─ artifacts/                   # CNN 训练输出：模型、metadata、scaler
├─ artifacts_mlp/               # MLP 训练输出：模型、metadata、scaler
├─ data/processed/              # 预处理后的联邦数据
├─ flows.csv / 1.csv            # 示例或中间数据文件
└─ README.md
```

## 图形界面

本项目已新增一个基于 `Streamlit` 的网页控制台 `app.py`，可以在浏览器里直接操作主要功能：

- 数据预处理
- CNN 联邦训练
- MLP 联邦训练
- CNN 模型评估
- MLP 模型评估
- 真实 CSV 推理

### 界面特点

- 现代化的网页布局，左侧选择功能，右侧填写参数
- 训练/预处理/推理任务均以后台方式执行
- 页面下方实时显示日志，方便观察运行过程
- 适合在 Windows 上直接使用，不需要自己再写命令行参数

### 启动方式

```bash
streamlit run app.py
```

如果你想用更完整的环境，也可以先安装依赖后再启动界面：

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 完整功能概览

### 1. 数据预处理

核心脚本是 `scripts/prepare_cicids2017.py`，它负责：

- 从 CICIDS2017 `MachineLearningCSV` 目录读取一个或多个 CSV
- 去掉非数值列，清洗无穷值和空值
- 对标签做 `LabelEncoder` 编码
- 可选地：
  - 按每个 CSV 抽样 `sample-per-file`
  - 留出独立测试集 `--test-fraction`
  - 对 `BENIGN` 类做下采样 `--benign-max-ratio`
  - 使用 `--demo` 生成合成数据调试流程
- 使用 `StandardScaler` 只在训练部分拟合标准化参数
- 将训练集按 `StratifiedKFold(10)` 划分为 10 个客户端
- 输出：
  - `client_0.npz` ~ `client_9.npz`
  - `metadata.json`
  - `global_scaler.npz`
  - 可选 `test.npz`

### 2. 联邦训练

项目提供两套训练脚本：

#### `federated/train_fl.py`
CNN 版本的联邦训练。

- 模型结构：二维卷积网络 + 全连接层 + softmax 输出
- 输入：把标准化后的特征重排为灰度图
- 支持两种训练路径：
  - 默认 Keras FedAvg 模拟
  - 通过环境变量 `FL_USE_TFF=1` 切换到 TensorFlow Federated
- 支持：
  - `crossentropy` / `focal loss`
  - `class_weight`
  - `dropout`
  - 可配置隐藏层宽度
  - 客户端优化器 `sgd` / `adam`
  - 每轮学习率衰减
  - 梯度裁剪 `clipnorm`
- 训练完成后导出：
  - `global_model.keras`
  - `metadata.json`
  - `global_scaler.npz`

#### `federated/train_fl_mlp.py`
MLP 对比版本的联邦训练。

- 输入直接使用一维特征向量
- 其他训练能力与 CNN 版基本一致
- 适合做结构对比实验，查看 CNN 与 MLP 在同一联邦划分下的表现差异
- 输出目录默认是 `artifacts_mlp`

### 3. 推理与评估

#### `federated/inference.py`
CNN 模型推理与评估。

支持：

- 单独做 smoke test
- 读取 `NPZ` 数据评估多分类准确率
- 输出：
  - 多分类 `classification_report`
  - 多分类召回率 `micro / macro / weighted`
  - 异常检测混淆统计（TN / FP / FN / TP）
  - 可选完整混淆矩阵
  - 可选诊断输出，分析模型是否退化为“全 BENIGN”
- 异常分数定义：`1 - P(BENIGN)`
- 默认判定规则：`score >= threshold` 则视为异常

#### `federated/inference_mlp.py`
MLP 模型推理与评估。

- 功能与 CNN 版一致
- 仅适配一维输入特征
- 默认读取 `artifacts_mlp`

#### `federated/inference_real_csv.py`
真实 CSV 逐行预测脚本。

- 面向 `pcap_to_simple_flow_csv.py` 产出的流量 CSV
- 读取训练得到的 `metadata.json` 和 `global_scaler.npz`
- 自动对齐特征列、处理别名列名、标准化并加载模型
- 支持：
  - 输出 top-k 预测类别
  - 显示概率
  - 限制只预测前 N 行
- 适合部署、联调和真实流量预测

### 4. CSV / NPZ 转换

#### `scripts/csv_to_test_npz.py`
将 flow CSV 转为可测试的 `test.npz`。

- 读取训练元数据并对齐特征列
- 使用训练期保存的 scaler 进行标准化
- 支持标签映射、覆盖标签、忽略标签
- 生成适合当前模型的 `x, y` 测试包

#### `scripts/pcap_to_simple_flow_csv.py`
从名称可判断，这是将 PCAP 转为简化流特征 CSV 的预处理脚本入口。

#### `scripts/prepare_cicids2017_train_split.py`
从名称和用途看，它是另一套基于 CICIDS2017 的训练集划分/预处理脚本，属于历史版本或扩展版本，和当前 `prepare_cicids2017.py` 一起构成数据准备相关功能。

## 数据流转关系

```text
原始 CICIDS2017 CSV / PCAP
        ↓
scripts/pcap_to_simple_flow_csv.py（可选）
        ↓
scripts/prepare_cicids2017.py
        ↓
data/processed/
  ├─ client_0.npz ... client_9.npz
  ├─ metadata.json
  ├─ global_scaler.npz
  └─ test.npz（可选）
        ↓
federated/train_fl.py 或 federated/train_fl_mlp.py
        ↓
artifacts/ 或 artifacts_mlp/
  ├─ global_model.keras
  ├─ metadata.json
  └─ global_scaler.npz
        ↓
federated/inference.py / federated/inference_mlp.py / federated/inference_real_csv.py
```

## 关键文件说明

### `metadata.json`
保存训练和推理一致性所需的关键信息：

- `n_clients`
- `n_features`
- `n_classes`
- `label_classes`
- `feature_names`
- `iid` 划分方式
- `min_samples_per_class`
- `test_split` 配置
- `rebalance` 配置

### `global_scaler.npz`
保存标准化所需的：

- `mean`
- `scale`

推理阶段必须使用与训练一致的 scaler。

### `global_model.keras`
联邦训练后的全局模型文件。

## 常见使用流程

### 1. 预处理数据

```bash
python scripts/prepare_cicids2017.py --csv-dir MachineLearningCSV --output-dir data/processed
```

如果想保留测试集：

```bash
python scripts/prepare_cicids2017.py --csv-dir MachineLearningCSV --test-fraction 0.2 --output-dir data/processed
```

### 2. 训练 CNN 联邦模型

```bash
python federated/train_fl.py --data-dir data/processed --artifacts-dir artifacts
```

### 3. 训练 MLP 联邦模型

```bash
python federated/train_fl_mlp.py --data-dir data/processed --artifacts-dir artifacts_mlp
```

### 4. 评估模型

CNN：

```bash
python federated/inference.py --artifacts-dir artifacts --eval-npz data/processed/test.npz
```

MLP：

```bash
python federated/inference_mlp.py --artifacts-dir artifacts_mlp --eval-npz data/processed/test.npz
```

### 5. 对真实 CSV 做预测

```bash
python federated/inference_real_csv.py --csv your_flow.csv --artifacts-dir artifacts
```

## 训练与推理的设计特点

- 联邦训练默认采用 10 客户端固定划分
- 数据预处理和推理都严格依赖同一份 `metadata.json` 与 `global_scaler.npz`
- 推理脚本不会要求重新训练，只要 artifacts 完整即可运行
- 支持从异常检测视角输出二分类结果，同时保留多分类能力
- 提供 CNN / MLP 两种实验线，便于比较特征重排与直接向量输入的效果

## 注意事项

- `data/processed` 与 `artifacts` 必须来自同一次预处理/训练流程，否则可能出现类别数或特征顺序不一致
- 如果模型输出几乎总是 `BENIGN`，通常要检查：
  - 是否启用了 `class_weight`
  - 数据是否极度不均衡
  - 训练轮数是否太少
  - 是否正确使用了同一份 `metadata.json`
- Windows 控制台可能因编码导致乱码，推理脚本已显式切换为 UTF-8

## 依赖

项目主要依赖：

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `tensorflow-federated`（可选，使用 `FL_USE_TFF=1` 时需要）

## 简短总结

这是一个“CICIDS2017 流量预处理 + 联邦学习训练 + 多分类/异常检测推理”的完整实验项目，既可以做联邦入侵检测研究，也可以作为流量分类原型系统使用。
