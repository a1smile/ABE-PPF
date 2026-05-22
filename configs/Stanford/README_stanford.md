1. # Stanford 配置说明（修订版）

   本目录提供 Stanford 数据集实验所需的配置文件。

   本次修订的核心目标是：

   1. 明确不同方法的真实执行路径
   2. 避免将 backend fallback 路径混入 front-end 对照实验
   3. 区分 `ppf_frontend_time`、`backend_time` 和 `registration_time`
   4. 使消融实验定义更加清晰、可追踪、可解释

   除特别说明的配置外，各实验尽量保持相同的预处理、采样和几何参数，仅对目标模块进行开启或关闭，或显式指定最终姿态策略。

   ---

   ## 1. 方法定义

   ### 1.1 Original Baseline

   原始 PPF 风格配置，不启用以下模块：

   - RS-MRQ
   - Robust Vote
   - Backend hypothesis consolidation

   同时，最终姿态采用原始风格的 legacy clustering 路径：

   - `final_pose_policy: legacy_cluster`

   对应配置：

   - `baseline_stanford.yaml`

   > 说明：  
   > 该配置用于回答“相对于原始系统，完整方法整体提升了多少”这一问题。  
   > 它不应被解释为“与完整方法仅在三个创新点开关上完全一致”的严格同骨架对照。

   ---

   ### 1.2 Same-backbone Baseline

   该配置不启用：

   - RS-MRQ
   - Robust Vote
   - Backend hypothesis consolidation

   但为了避免退回 legacy clustering 慢路径，最终姿态显式由原始候选中的 top-1 vote 给出：

   - `final_pose_policy: raw_top1_vote`

   对应配置：

   - `same_backbone_baseline_stanford.yaml`

   > 说明：  
   > 该配置用于回答“在相同执行骨架下，关闭创新模块后性能如何变化”这一问题。  
   > 它比 Original Baseline 更适合做公平的速度和路径对比。

   ---

   ### 1.3 Front-end only

   仅启用前端增强：

   - RS-MRQ
   - Robust Vote

   不启用 backend hypothesis consolidation，即不使用：

   - 多指标 pose selection
   - candidate veto
   - light refine
   - pose clustering

   最终姿态显式由 top-1 candidate 给出：

   - `final_pose_policy: selected_top1`

   对应配置：

   - `ablation_no_pose_pipeline_stanford.yaml`

   > 说明：  
   > 这里的 “Front-end only” 含义是：  
   > **启用召回增强与鲁棒投票，但不使用后端多假设整合能力。**
   >
   > 这不是简单地关闭 `pose_selection` 入口，而是通过显式 `final_pose_policy` 避免隐式回退到旧的 legacy clustering 路径。  
   > 因此，它应被理解为  
   > **without backend hypothesis consolidation 的安全实现版本**，  
   > 而不是严格意义上的“只关一个布尔开关”。

   ---

   ### 1.4 Strict No Mode Cluster

   该配置保留：

   - RS-MRQ
   - Robust Vote
   - pose selection
   - candidate veto
   - light refine

   仅移除：

   - pose clustering

   最终姿态由经过候选级评分后的 top-1 hypothesis 给出：

   - `final_pose_policy: selected_top1`

   对应配置：

   - `strict_no_mode_cluster_stanford.yaml`

   > 说明：  
   > 该配置用于更严格地分析 mode clustering 本身的贡献。

   ---

   ### 1.5 Ours (Full)

   完整方法，启用：

   - RS-MRQ
   - Robust Vote
   - Backend hypothesis consolidation

   其中 backend hypothesis consolidation 包括：

   - pose selection
   - candidate veto
   - top-k light refine
   - pose clustering

   最终姿态采用：

   - `final_pose_policy: selected_plus_mode_cluster`

   对应配置：

   - `ablation_ours_stanford.yaml`

   ---

   ### 1.6 其它消融

   #### No RS-MRQ

   关闭 RS-MRQ，其它与完整方法保持一致。

   对应配置：

   - `ablation_no_rsmrq_stanford.yaml`

   #### No Robust Vote

   关闭 Robust Vote，其它与完整方法保持一致。

   对应配置：

   - `ablation_no_robust_vote_stanford.yaml`

   ---

   ## 2. 推荐实验分组

   建议最终论文或报告按以下分组展示：

   | Method                 | Description                                                  |
   | ---------------------- | ------------------------------------------------------------ |
   | Original Baseline      | Original PPF with legacy final clustering                    |
   | Same-backbone Baseline | Baseline front-end under the new execution scaffold          |
   | No RS-MRQ              | Full method without RS-MRQ                                   |
   | No Robust Vote         | Full method without robust voting                            |
   | Front-end only         | RS-MRQ + Robust Vote, final pose selected by top-1 candidate |
   | Strict No Mode Cluster | Full method without mode clustering                          |
   | Ours (Full)            | RS-MRQ + Robust Vote + backend hypothesis consolidation      |

   ---

   ## 3. 时间指标说明

   为避免将不同阶段的时间混在一起误解释，本实验区分以下三类时间指标。

   ### 3.1 `ppf_frontend_time`

   表示前端阶段时间，主要包括：

   - 候选生成
   - 特征检索
   - 投票累积
   - 相关前端匹配过程

   该指标更接近 front-end matching / voting 开销。

   ---

   ### 3.2 `backend_time`

   表示后端阶段时间，主要包括：

   - pose selection
   - candidate veto
   - light refine
   - pose clustering
   - legacy clustering（若触发）

   该指标用于分析最终姿态决策阶段的额外开销。

   ---

   ### 3.3 `registration_time`

   表示当前实现下的总注册时间，通常包括：

   - scene preprocess
   - front-end
   - backend

   因此，`registration_time` 是整体系统开销，不应直接等同于纯 matching time。

   ---

   ### 3.4 推荐报告方式

   建议在实验中分别报告：

   - `ppf_frontend_time`
   - `backend_time`
   - `registration_time`

   其中：

   - 若讨论前端效率，请优先使用 `ppf_frontend_time`
   - 若讨论系统总开销，请使用 `registration_time`
   - 若分析后处理代价，请使用 `backend_time`

   > 说明：  
   > 不建议再将 `registration_time` 直接表述为“matching time”。

   ---

   ## 4. 运行命令

   以下命令默认在项目根目录执行。

   ### 4.1 Original Baseline

   ```bash
   python scripts/run_batch_stanford.py --config configs/Stanford/baseline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_baseline --num_workers 8 --rebuild_cache

> 需要重建 cache，因为第一次运行或模型结构可能尚未缓存。

### 4.2 Same-backbone Baseline

```
python scripts/run_batch_stanford.py --config configs/Stanford/same_backbone_baseline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_same_backbone_baseline --num_workers 8 --rebuild_cache
```

> 需要重建 cache，因为该配置关闭了 RS-MRQ。

### 4.3 No RS-MRQ

```
python scripts/run_batch_stanford.py --config configs/Stanford/ablation_no_rsmrq_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_rsmrq --num_workers 8 --rebuild_cache
```

> 需要重建 cache，因为模型检索结构发生变化。

### 4.4 No Robust Vote

```
python scripts/run_batch_stanford.py --config configs/Stanford/ablation_no_robust_vote_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_no_robust_vote --num_workers 8
```

> 不需要重建 cache。

### 4.5 Front-end only

```
python scripts/run_batch_stanford.py --config configs/Stanford/ablation_no_pose_pipeline_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_frontend_only --num_workers 8
```

> 不需要重建 cache。
>  该配置采用显式 top-1 final pose policy，不会退回 legacy clustering。

### 4.6 Strict No Mode Cluster

```
python scripts/run_batch_stanford.py --config configs/Stanford/strict_no_mode_cluster_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_strict_no_mode_cluster --num_workers 8
```

> 不需要重建 cache。

### 4.7 Ours (Full)

```
python scripts/run_batch_stanford.py --config configs/Stanford/ablation_ours_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_ours_full --num_workers 8
```

> 不需要重建 cache。

### 4.8 Debug Fast

```
python scripts/run_batch_stanford.py --config configs/Stanford/debug_fast_stanford.yaml --csv stanford_retrieval_batch_all_variants.csv --cache_dir data/stanford_bunny_ppf/model_cache --out_prefix stanford_debug_fast --num_workers 8
```

> 用于快速检查流程是否正常，不用于最终结果汇报。

------

## 5. 关于 cache 的说明

以下情况需要使用 `--rebuild_cache`：

1. 第一次运行 Stanford 实验
2. 修改了模型特征相关配置，例如：
   - `enable_rsmrq`
   - `sampling_leaf`
   - `angle_step_deg`
   - `distance_step_ratio`
   - 其它会影响模型哈希或检索结构的参数

以下情况通常不需要重建 cache：

- 只修改 robust vote 参数
- 只修改 pose selection / veto / refine / clustering 参数
- 只修改 `final_pose_policy`
- 只修改输出目录或可视化参数

------

## 6. 结果目录建议

建议结果按如下方式保存，避免新旧实验混淆：

```
experiments/results/
  stanford_baseline/
  stanford_same_backbone_baseline/
  stanford_no_rsmrq/
  stanford_no_robust_vote/
  stanford_frontend_only/
  stanford_strict_no_mode_cluster/
  stanford_ours_full/
  stanford_debug_fast/
```

------

## 7. 结果解释建议

### 7.1 关于 Original Baseline

Original Baseline 的总注册时间可能显著偏大。
 这不仅因为它缺少 RS-MRQ 和 Robust Vote，还可能因为最终姿态会经过 legacy clustering 路径。

因此：

- 若讨论“相对原始系统的整体提升”，可使用 Original Baseline
- 若讨论“在同一执行骨架下的公平速度对比”，应优先参考 Same-backbone Baseline

------

### 7.2 关于 RS-MRQ

RS-MRQ 的主要作用是：

- 提升候选召回
- 提升检索稳定性
- 为后续 robust vote 提供更稳定的候选基础

其对总运行时间的影响，应结合：

- `ppf_frontend_time`
- `backend_time`
- `registration_time`

共同分析。

> 说明：
>  不建议在未拆分时间项的前提下，直接将总时间变化完全归因于 RS-MRQ。

------

### 7.3 关于 Robust Vote

Robust Vote 的主要作用是：

- 抑制噪声候选
- 减少虚假高峰
- 改善投票稳定性

它通常对 hardest cases 更重要，也可能显著影响尾部样本的时间与稳定性。

------

### 7.4 关于 Backend Hypothesis Consolidation

Backend hypothesis consolidation 的主要作用是：

- 提升候选决策质量
- 解决多峰歧义
- 从多个局部高分候选中恢复更可信的 pose mode
- 提高 hardest cases 的鲁棒性

其中可进一步拆分为：

- candidate-level scoring
- candidate veto
- top-k light refine
- mode clustering

------

## 8. 建议的核心结论组织方式

建议将 Stanford 消融的核心结论组织为：

1. **RS-MRQ** 主要提升候选召回与检索稳定性
2. **Robust Vote** 主要提升证据聚合质量并抑制错误峰值
3. **Backend hypothesis consolidation** 主要提升最终姿态决策稳定性与多峰歧义处理能力
4. 不同模块对速度的贡献应结合
   - `ppf_frontend_time`
   - `backend_time`
   - `registration_time`
      共同分析

------

## 9. 备注

- `Front-end only` 是一个安全定义的 backend-removed 版本，不是简单关闭某个入口开关。
- `Original Baseline` 与 `Same-backbone Baseline` 的作用不同，建议不要混为一谈。
- 若论文中讨论“matching efficiency”，应优先引用 `ppf_frontend_time`，而不是直接使用 `registration_time`。
- 若论文中讨论“system efficiency”，则可使用 `registration_time`，但应明确其包含 backend 开销。