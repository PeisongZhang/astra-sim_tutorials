# chakra / STG 检查报告

## 结论

**结论分两层：**

1. **STG 最近两次修改本身没有把 STG 生成功能搞坏。**  
   相关功能路径可以正常生成 Llama3-8B workload，且新增的定向测试通过。

2. **`llama3_8b.sh` 最新修改对“生成 LocalSGD 风格的 DNN workload”是正确的，但它已经不再等价于之前的“默认单 iteration、同步 DP workload”脚本。**  
   也就是说：  
   - **如果目标是 LocalSGD workload**：这次修改是对的。  
   - **如果目标还是旧的同步 DP workload**：这次修改改变了默认语义，不应该再把它当成旧脚本来用。

---

## 检查范围

- 脚本：`micro2024/chakra-demo/demo3/llama3_8b.sh`
- STG 实际仓库：`micro2024/symbolic_tensor_graph`
- 说明：`/home/ps/sow/part2/micro2024_tutorial` 实际是指向 `astra-sim_tutorials/micro2024` 的符号链接，所以脚本真实使用的是当前仓库里的这份 STG。

---

## 检查的提交

### `llama3_8b.sh` 最近两次提交

| 提交 | 含义 | 关键变化 |
| --- | --- | --- |
| `5bb4a42` | `batch` | `--micro_batch` 从 `2` 改成 `8` |
| `33e5fbf` | `local sgd` | 增加 `NUM_ITERATIONS=8`、`DP_LOCAL_SGD_INTERVAL=8`，并透传到 `main.py` |

### STG 最近两次提交

| 提交 | 含义 | 关键变化 |
| --- | --- | --- |
| `9f28f62` | `batch` | `MicroBatchReplicator` 改成先合并本地梯度，再在 step 级做一次 DP 同步 |
| `946da0a` | `local sgd` | 增加 `--num_iterations` / `--dp_local_sgd_interval`，并新增 `LocalSGDIterationPostProcess` |

---

## 代码层面核查

### 1. `batch` 这次修改是不是正确

**是正确的。**

`llama3_8b.sh` 中当前参数是：

- `dp=4`
- `pp=4`
- `batch=128`
- `micro_batch=8`

而 STG 当前实现里，`micro_batch` 的语义不是“每个 DP rank 的 local micro-batch”，而更接近**DP 切分前的 micro-batch 大小**。  
因此在 `dp=4` 时：

- `micro_batch=8`
- 对应每个 DP rank 的 local micro-batch = `8 / 4 = 2`

这和脚本注释里想表达的“`local_micro_batch(2)`”是一致的。  
所以 **`5bb4a42` 把 `--micro_batch 2` 改成 `8` 是正确修正，不是错误修改。**

同时，当前 STG 的 micro-batch 数按：

- `batch / micro_batch = 128 / 8 = 16`

来展开。实际生成的 ET 中也确实看到了：

- `mb0` 到 `mb15`
- 共 **16 个 micro-batch**

这说明脚本参数和 STG 当前语义是对齐的。

### 2. STG 的 `batch` 修改有没有破坏功能

**没有发现破坏。**

`9f28f62` 主要改的是 `symbolic_tensor_graph/graph/grad_updater.py` 里的 `MicroBatchReplicator.apply(...)`：

- 新增 `_should_merge_before_sync(...)`
- 新增 `_create_merged_grad(...)`
- 对“先同步再更新”的梯度路径做了重构

核心效果是：

- 每个 `mb*` 的梯度先本地合并
- 然后只在 **step 级** 触发一次 DP collective

这和常见训练语义更接近，也和脚本当前配置匹配。  
对应新增测试 `test_cases/symbolic_tensor_graph/graph/test_grad_updater.py` 里，验证了：

- micro-batch 复制后先合并本地梯度
- 最终只产生一次 step 级 `ALL_REDUCE`

### 3. STG 的 `local sgd` 修改有没有破坏功能

**也没有发现破坏。**

`946da0a` 主要做了两件事：

1. 在 `main.py` 中新增：
   - `--num_iterations`
   - `--dp_local_sgd_interval`
2. 在 `grad_updater.py` 中新增 `LocalSGDIterationPostProcess`

这个后处理做的事情是：

- 把单 iteration 图复制成多个 iteration
- 在非同步 iteration 上删除 DP `ALL_REDUCE`
- 重写依赖关系
- 插入 barrier 把 iteration 串起来
- 为点对点通信偏移 `comm_tag`

从实现方式看，它是**在 Chakra 图后处理阶段做增量改动**，没有推翻原有生成主链路；默认参数仍然是：

- `num_iterations=1`
- `dp_local_sgd_interval=1`

因此 **STG 默认同步 DP 行为没有被破坏**；只有显式传入更大的参数时，才会切换成 LocalSGD 风格。

---

## 实际运行结果

我没有直接运行脚本本身，以免覆盖仓库里的 `llama/` 输出；而是用**与脚本相同参数**直接调用 `main.py`，输出到临时目录进行检查。

使用参数：

```bash
python3 main.py \
  --output_dir <temp> \
  --output_name llama.%d.et \
  --dp 4 --tp 1 --pp 4 \
  --seq 8192 --batch 128 \
  --dvocal 128256 --dmodel 4096 --dff 14336 \
  --head 32 --kvhead 8 --num_stacks 32 \
  --micro_batch 8 \
  --num_iterations 8 \
  --dp_local_sgd_interval 8 \
  --model_type llama \
  --mixed_precision true \
  --weight_sharded 0
```

### 生成结果

- 成功生成 **16 个** `.et` 文件
- 同时生成 `llama.json`
- 这和 `dp * pp = 4 * 4 = 16` 一致

### ET 抽样检查结果

以 `llama.0.et` 为例，解析出的关键信息是：

- 共 **8 个 iteration**
- 共 **16 个 micro-batch**（`mb0` 到 `mb15`）
- 共 **7 个 barrier**（`iter0_to_iter1_BARRIER` 到 `iter6_to_iter7_BARRIER`）
- 所有 collective 都只出现在 **`iter7`**
- **没有 `mb*` 级别的 collective**

这正好对应：

- `num_iterations = 8`
- `dp_local_sgd_interval = 8`

也就是：

- 前 7 个 iteration 只做本地训练
- 第 8 个 iteration 才做一次 DP 同步

对所有 rank 的 ET 做统计后，collective 也符合预期：

- rank `0-3`、`12-15`：每个 rank **9 个** collective
- rank `4-11`：每个 rank **8 个** collective

这和首尾 stage 含 embedding / output embedding 的差异一致，不像是依赖或通信图被破坏后的异常结果。

---

## 测试结果

### 与本次修改直接相关的测试

运行：

```bash
python3 -m unittest test_cases.symbolic_tensor_graph.graph.test_grad_updater
```

结果：

- **6 个测试全部通过**

覆盖点包括：

- step 级 collective 数量
- LocalSGD iteration 复制
- 非同步 iteration 上 DP all-reduce 删除后的依赖重连
- point-to-point `comm_tag` 偏移

### 仓库内其它测试

在 `test_cases/symbolic_tensor_graph` 子树做发现式运行时，有 **1 个失败**：

- `test_tensor.TestTensor.test_op_handler3`
- 失败内容是 `[1.00000000000000]` vs `[1]`

这个失败对应的文件不在最近两次 STG 提交改动范围内，因此**不能归因于这次 `batch` / `local sgd` 修改**。  
更像是仓库里原本就存在的数值表示差异问题。

---

## 是否“破坏了 STG 功能”

**结论：没有。**

更准确地说：

- **STG 原有同步 DP 生成功能没有被破坏**
- **新增 LocalSGD workload 生成功能可以正常工作**
- **当前脚本参数与最新 STG 语义匹配，能正确生成目标 workload**

我没有看到下面这些“功能被破坏”的信号：

- 参数无法解析
- 生成流程报错
- rank 数量错误
- iteration 数量错误
- micro-batch 数错误
- non-sync iteration 仍然残留 DP all-reduce
- barrier 缺失导致 iteration 没串起来
- point-to-point tag 冲突

这些都没有出现。

---

## 但脚本层面有两个明显风险点

### 风险 1：脚本默认语义已经变了

`33e5fbf` 把：

- `NUM_ITERATIONS=8`
- `DP_LOCAL_SGD_INTERVAL=8`

直接写死了。

这意味着 `llama3_8b.sh` **默认不再生成旧的同步 DP 单 iteration workload**，而是默认生成：

- 8 个连续 iteration
- 只在最后一个 iteration 做 DP 同步

所以如果有人还把这个脚本当作“普通 Llama3-8B 同步 DP workload 生成器”来用，**那就不对了**。

### 风险 2：输出目录逻辑被注释掉，容易误导

脚本里原本打算在开启 LocalSGD 时切到：

- `llama_local_sgd/`

但相关逻辑现在被注释掉了，最终固定成：

- `OUTPUT_DIR=${SCRIPT_DIR}/llama/`

这会带来两个问题：

1. **语义误导**：目录名还是 `llama/`，但里面其实是 LocalSGD trace  
2. **覆盖风险**：可能覆盖之前的同步 DP 输出

而且 `local_sgd.md` 里写的是“脚本会在开启 LocalSGD 时自动输出到 `llama_local_sgd/`”，当前脚本实现已经和这份说明不一致。

---

## 最终判断

### 关于 “STG 最近两次修改是否破坏功能”

**否。**  
从代码、测试、和实际生成结果看，最近两次 STG 修改没有破坏 STG 的核心 workload 生成功能，反而补齐了：

- step 级 DP 同步语义
- 多 iteration LocalSGD workload 生成能力

### 关于 “最新修改对生成 DNN workload 是否正确”

**有条件地是正确的：**

- **对 LocalSGD workload 而言：正确**
  - `micro_batch=8` 与当前 STG 语义匹配
  - `num_iterations=8`
  - `dp_local_sgd_interval=8`
  - 生成结果与参数完全一致

- **对旧的默认同步 DNN workload 而言：不再正确**
  - 因为脚本默认语义已经改成 LocalSGD
  - 且输出目录没有显式区分

---

## 建议

如果这个脚本的目标确实是 **LocalSGD 专用生成器**，那当前 STG 路径没有问题；  
但建议至少把脚本层面改成下面二选一中的一种：

1. **恢复环境变量/参数化**  
   让 `NUM_ITERATIONS` 和 `DP_LOCAL_SGD_INTERVAL` 可覆盖；

2. **至少恢复输出目录分流**  
   开启 LocalSGD 时输出到 `llama_local_sgd/`，避免把 LocalSGD trace 混进 `llama/`。

否则使用者很容易把当前输出误认为普通同步 DP workload。
