# `llama3_8b.sh` 生成 DP workload 与 LocalSGD 的分析

## 结论

`llama3_8b.sh` 目前只能生成**同步 DP**（梯度按当前图规则进行 `ALL_REDUCE`）的 workload，不能直接通过命令行参数切换到 LocalSGD。  
要生成 LocalSGD DP workload，需要额外加入“**每 K 个 local step 才做一次全局同步**”的逻辑（推荐 ET 后处理，或在 STG 中增加原生参数）。

---

## 1) 当前脚本如何生成 DP workload

`llama3_8b.sh` 关键调用（见 `llama3_8b.sh`）：

- `python3 main.py`
- `--dp 4 --tp 1 --pp 4`
- `--seq 8192 --batch 128 --micro_batch 2`
- `--model_type llama --weight_sharded 0`

这会在 `llama/` 下生成每个 rank 的 Chakra ET（如 `llama.0.et`）和通信组文件 `llama.json`。

在 STG 主流程（`symbolic_tensor_graph/main.py`）中，执行顺序是：

1. 组装模型图（llama）
2. `MicroBatchReplicator.apply(...)`
3. `GradUpdater.apply(...)`
4. `GraphDistributer.apply(...)`
5. `BundledConvertChakra.apply(...)` 输出 ET

---

## 2) 为什么默认是同步 DP（不是 LocalSGD）

1. `main.py` 的参数列表里没有 `local_sgd` / `sync_interval` 一类参数（只有 `dp/tp/pp/sp/...`）。  
2. `convert_chakra.py` 会根据张量并行语义自动插入通信节点。  
3. `coll_comm_matcher.py` 的规则里，`PARTIALSUM -> DUPLICATED` 会映射到 `ALL_REDUCE`，这正是同步 DP 梯度聚合语义。  
4. 抽样读取 `llama/llama.0.et` 后可见 collective 节点均为 `ALL_REDUCE`，并且按 `mb0..mb63` 重复出现（和 `batch=128, micro_batch=2` 对应）。

因此：当前链路会周期性地在图中保留 DP all-reduce，同步语义是“内生的”，不是运行时可切换的 LocalSGD。

### 补充：这里的 `micro_batch` 语义与常见 PP 训练定义不一致

这次进一步核查后，可以更明确地说：**当前 STG / Chakra workload 中的 `micro_batch`，并不是多维并行训练里常说的“PP 调度 micro-batch”语义。**

在常见的 DP+PP 训练定义里：

- `batch` 通常表示 **global batch size**；
- `micro_batch` 通常表示 **每个 pipeline slot / 每个 rank 一次前反向处理的样本数**；
- 因此一个 optimizer step 内的 PP micro-batch 数通常应为：

  `num_micro_batches = batch / (dp * micro_batch)`

对本例 `dp=4, batch=128, micro_batch=2`，若按这个标准定义，应得到：

- `128 / (4 * 2) = 16` 个 PP micro-batches。

但当前 STG 实现并不是这样算的。`main.py` 只是把 `Batch=args.batch`、`MicroBatch=args.micro_batch` 原样塞进符号表；随后 `MicroBatchReplicator.apply(...)` 直接用：

- `num_batches = Batch / MicroBatch`

来复制图，并生成 `mb0`, `mb1`, ... 这些前缀。也就是说，本例实际被展开成：

- `128 / 2 = 64` 个 `mb*`

而**没有除以 `dp`**。这与标准 PP micro-batch 计数方式不同。

进一步地，梯度通信插入逻辑会把 `PARTIALSUM -> DUPLICATED` 映射为 `ALL_REDUCE`，而 `convert_chakra.py` 会为每个匹配到的 tensor 生成 `COMM_COLL_NODE`。因此 ET 中看到的 `mb0..mb63` 上的 `_sharded_grad@..._X1COMM`，表示的是：

- 每个被复制出的 `mb*` 局部训练块，都会触发一次 DP `ALL_REDUCE`；
- 而不是“多个 PP micro-batches 先做梯度累积，在 step 边界统一做一次 DP 同步”。

所以，从建模语义上更准确的说法是：

- 这里的 `mb*` 更接近 **被 STG 显式复制出来的 local training chunk / local step**；
- 不是标准 1F1B / GPipe 文献语境下、服务于 PP 调度和梯度累积的那种 micro-batch。

这也是为什么当前 workload 不适合直接拿来表示 LocalSGD：它默认假设**每个 `mb*` 结束就进行一次同步 DP 梯度聚合**。

---

## 3) 怎么生成 LocalSGD DP workload（可行方案）

### 方案 A（推荐，侵入最小）：先生成同步 DP，再做 ET 后处理

流程：

1. 先运行原 `llama3_8b.sh`，得到 baseline ET。  
2. 写一个后处理脚本（例如 `local_sgd_postprocess.py`）读取 `llama.%d.et`：  
   - 识别 DP 的 `COMM_COLL_NODE + ALL_REDUCE`；  
   - 设 LocalSGD 周期 `K`；  
   - 若 `(local_step + 1) % K != 0`，将该 all-reduce 改为“本地步不全局同步”（常见做法：改到 singleton group）；  
   - 仅在第 `K` 步保留原始 DP all-reduce。  
3. 输出新文件到 `llama_local_sgd/`，避免覆盖原 traces。

> 对本例（`dp=4,tp=1,pp=4`）而言，`llama.json` 中 size>1 的组是 DP 组，single-rank 组可用于“本地步 no-op 通信组”。

### 方案 B（长期更干净）：在 STG 增加原生 LocalSGD 参数

建议改造点：

1. 在 `main.py` 新增参数：`--dp_local_sgd_interval`（默认 `1`，保持现有行为）。  
2. 将该参数传入 `ConvertChakra/BundledConvertChakra`。  
3. 在 `convert_chakra.py::_insert_comm_x1/_insert_comm_x2` 中，仅对 `parallel_dim == dp` 的 `ALL_REDUCE` 应用周期逻辑：  
   - 非同步步：不插入 DP collective（或改到 singleton group）；  
   - 同步步：保留现有 all-reduce 插入逻辑。  

这样可直接从生成器产出 LocalSGD trace，不依赖额外后处理。

---

## 4) 对 `llama3_8b.sh` 的落地建议

如果目标是“尽快得到 LocalSGD workload”，优先走**方案 A（ET 后处理）**：  
保留当前脚本生成 baseline，再用 `LOCAL_SGD_K` 控制后处理同步周期，生成 `llama_local_sgd/` 版本 traces。  
若后续会反复做策略扫描，再考虑落地方案 B，把 LocalSGD 变成 `main.py` 原生参数。
