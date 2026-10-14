# Reward 配置说明

所有可调评分参数集中在 [`configs/reward_config.yaml`](configs/reward_config.yaml)，VG/PSG 关系频率表在 [`configs/relation_frequencies.yaml`](configs/relation_frequencies.yaml)。修改参数后重启训练生效，不会在训练中自动热更新。

## 使用方式

```bash
REWARD_CONFIG_PATH="src/rl/configs/reward_config.yaml" bash scripts/rl/run_sgg_rl.sh
```

也可以向 Python 入口传入 `--reward_config_path /path/to/experiment.yaml`。实验 YAML 支持只填写需要覆盖的项，其余值继承项目默认配置；未知字段、非法类型、负数、非有限数和关键除数为零会在初始化时报错。

```yaml
# 一个局部覆盖示例
weights:
  format: 0.5
  edge: 2.0
stage2_node:
  recall:
    iou_threshold: 0.6
stage3_edge:
  coarse:
    match_threshold: 0.8
```

没有显式指定 `frequencies_path` 的局部配置使用项目内置频率表；显式指定的相对路径以该实验 YAML 所在目录为基准。复制完整配置到别处时，也需要复制频率表或更改路径。debug 日志相对路径则以运行工作目录为基准。

## 总奖励

回调顺序固定为 format、category、node、edge，分别对应格式奖励、阶段一类别奖励、阶段二节点奖励、阶段三关系奖励。

```text
R = weights.format * R_format
  + weights.category * R_category
  + weights.node * R_node
  + weights.edge * R_edge
```

四个默认总权重都是 1.0，训练入口把它们传给 TRL 的 `reward_weights`，回调自身返回未乘总权重的分数，避免重复加权。某项设为 0 可排除其对总奖励的贡献，但仍计算并记录该回调。至少有一项总权重大于零。不要另外传入不同的 `--reward_weights`。

## 格式奖励：format_reward

分别检查 CATEGORY、OBJECT、RELATION 三个标签内的 JSON，然后按 `stage_weights.category/object/relation`（均默认 1）相加。

| scores 参数 | 默认值 | 条件 |
| --- | --- | --- |
| missing_tag | 0.0 | 缺少对应标签 |
| invalid_json | 0.5 | 标签存在，JSON 解析失败 |
| invalid_schema | 0.5 | JSON 顶层结构、必需字段或关系分组不正确 |
| invalid_item | 0.75 | 类别、对象或关系列表中的条目不合法 |
| valid_stage | 1.0 | 该阶段格式检查通过 |

默认总分范围为 0–3。这里只检查格式，不检查预测内容是否正确；空列表是否通过沿用现有结构检查规则。关系分组由 `dataset.type` 决定，必须与训练数据类型一致。

## 阶段一：stage1_category

以去重后的类别集合计算 precision、recall 和 F1：`2 * P * R / max(epsilon, P + R)`。类别完全相同时默认得 1。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| duplicate_score | 0.0 | 预测出现重复类别时直接返回的分数 |
| error_score | 0.0 | 样本解析或计算异常时返回的分数 |

F1 中的系数 2 是公式定义，不是额外超参数；阶段贡献通过 `weights.category` 调节。

## 共用匹配：matching

阶段二和阶段三都使用匈牙利算法进行对象匹配，代价为：

```text
semantic_weight * (1 - category_similarity)
+ iou_weight * (1 - GIoU)
+ box_l1_weight * L1
```

三个权重默认分别为 1、3、3，至少一个大于零。这里 `iou_weight` 实际用于 GIoU，名称沿用原配置。`padding_cost=100000` 是预测对象不足时的补齐行代价，必须为正且应足够大；`cache_size=4096` 是匹配缓存容量，设为 0 禁用缓存。

`semantic_model.name_or_path` 默认 `all-MiniLM-L6-v2`，同时提供类别、三元组和谓词的语义向量。模型只在注册奖励时初始化，不再在导入 rewards 模块时重复加载。

## 阶段二：stage2_node

先匹配对象，再计算框分数和软召回，均以 GT 对象数量为分母，未匹配 GT 不加分：

```text
box_score = (box.iou_weight * IoU + box.l1_weight * exp(-box.l1_decay * L1))
            / (box.iou_weight + box.l1_weight)
R_node = box_weight * mean_box_score + recall_weight * soft_recall
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| box_weight / recall_weight | 1.0 / 1.0 | 框分数与软召回的组合系数 |
| box.iou_weight / box.l1_weight | 3.0 / 3.0 | 框分数内 IoU 与 L1 相似度的权重，和必须大于 0 |
| box.l1_decay | 1.0 | L1 指数衰减速度 |
| recall.iou_threshold | 0.5 | 定位匹配阈值，范围 [0, 1] |
| recall.full_match_score | 1.0 | 类别正确且 IoU 达标 |
| recall.partial_match_score | 0.5 | 类别正确或 IoU 达标，但不同时满足 |
| empty_score / error_score | 0.0 / 0.0 | 没有 GT/预测对象，或样本计算异常 |

类别和定位都不满足时加 0 分。默认范围为 0–2。框评分权重与 matching 的代价权重现在分开配置，调整匹配不再隐式改变框评分。

## 阶段三：stage3_edge

`R_edge = fine_weight * fine_reward + coarse_weight * coarse_reward`，两个组合系数默认均为 1；`error_score=0` 是样本计算异常的返回值。

细粒度奖励在对象匹配后，对每条 GT 关系选取候选中的最大 `三元组余弦相似度 × 谓词余弦相似度`，再按 GT 谓词长尾权重做加权平均。没有候选或没有 GT 关系时贡献 0。此处原本没有相似度阈值，仍使用连续分数，不新增阈值或改成加权相加；余弦乘积沿用原逻辑，不额外截断负值。

### 长尾权重：relation_weight

```text
log_inverse = log(1 / max(frequency, frequency_floor))
normalized = (log_inverse - log(1 / max_frequency))
             / max(log(1 / min_frequency) - log(1 / max_frequency), epsilon)
weight = round(base_weight + normalized * incremental_weight, rounding_digits)
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| base_weight | 1.0 | 频率表中最高频谓词的基础权重，必须大于 0 |
| incremental_weight | 2.0 | 稀有关系增加的权重幅度 |
| default_weight | 2.0 | 频率表未收录谓词的回退权重，保持旧奖励逻辑 |
| frequency_floor | 0.0001 | 单个频率的对数计算下限，必须大于 0 |
| rounding_digits | 2 | 权重保留小数位数，非负整数 |
| frequencies_path | relation_frequencies.yaml | 包含 vg/psg 映射的频率表 |

原始频率数值完整迁移，默认已收录关系权重约在 1–3。表内使用百分比口径，所有频率必须为正；谓词键要用引号，尤其是 `"on"`，防止 YAML 将其解释成布尔值。
frequency_floor 不得超过选中表的最小频率，以免归一化后产生负的增量。

### 粗粒度奖励：coarse

GT 三元组聚类后，把每条预测分配给相似度最高且达标的簇。覆盖率为已覆盖簇数 / 有效簇数，密度为匹配预测数 / 已覆盖簇内 GT 总数，并受 density_cap 限制。基础奖励等于覆盖率乘密度。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| cluster_threshold | 0.7 | 控制 DBSCAN 的 eps，范围 [0, 1) |
| match_threshold | 0.7 | 预测到簇中心的余弦相似度阈值，范围 [0, 1] |
| min_cluster_size | 1 | DBSCAN min_samples 及有效簇最小大小，正整数 |
| density_cap | 1.0 | 密度分数上限 |
| reward_cap | 1.0 | 粗粒度奖励最终上限，下限固定为 0 |
| imbalance_penalty.cv_threshold | 0.5 | 已覆盖簇密度变异系数 CV 的惩罚触发阈值 |
| imbalance_penalty.coefficient | 0.1 | CV 惩罚系数 |
| imbalance_penalty.max_penalty | 0.5 | 单次惩罚上限 |

覆盖多个簇且 CV 超过阈值时，减去 `min(max_penalty, CV * coefficient)`，结果不低于 0。默认粗粒度奖励在 0–1。

为了保持算法不变，保留原来的 `eps=sqrt(2*(1-cluster_threshold))` 与 DBSCAN cosine 距离组合，没有改成 `1-threshold`；只有一个 GT 三元组时原逻辑不构造簇中心，因此粗粒度贡献仍为 0。原先聚类与匹配共用同一个阈值，现在分别可调，默认仍都是 0.7。

## 迁移与边界

- 旧顶层 `coarse_reward` 移至 `stage3_edge.coarse`；旧 `relation_weight` 移至 `stage3_edge.relation_weight`。
- 旧 `debug.strict_format` 从未参与计算，已删除，不再保留无效开关。
- `numerics.epsilon=1e-9` 用于类别 F1 和长尾归一化的除零保护，不是业务阈值。
- 所有评分超参数默认值沿用原值。另修复关系格式检查把 dataset_type 固定为 vg 的问题：PSG 现在使用 PSG 的关系分组，因此该格式项可能与旧错误行为不同。
- 奖励训练路径中所有会改变奖励数值的权重、阈值、惩罚、回退分数、语义模型和匹配缓存容量均由 YAML 提供；`SemanticSimilarityManager` 的构造默认值仅为独立评估脚本保留，奖励注册时不会使用它们。
- 数学常量、集合计数、空匹配的零贡献以及解析格式仍保留在代码中，不作为调参项。
- 默认结构检查、重复类别惩罚、匹配、F1、细粒度相似度乘积与粗粒度覆盖算法保持不变。未进行 GPU 训练验证。

配置检查可在仅安装 PyYAML 的 CPU 环境运行：`python -B -m unittest discover -s tests -p test_reward_config.py`，不导入训练依赖或下载模型。
