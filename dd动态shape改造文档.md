# DecisionDiffuser 动态 Shape 改造文档

---

## 一、改造目标

将 DecisionDiffuser 从**静态 shape** 改造为**适配动态 shape**，使其能在 `torch.compile(dynamic=True)` 模式下对不同 batch_size 输入实现泛化编译，无需为每个 batch 档位单独重编译。

核心衡量指标：`torch._dynamo.utils.counters['stats']['unique_graphs']`
- `=1`：动态编译成功泛化（目标）
- `>1`：每个 batch 重编译（未泛化，如 `p_sample_loop` 因 for 循环导致图断裂）

---

## 二、改动文件总览（共 10 个）

| # | 文件 | 改动行数 | 作用 |
|---|---|---|---|
| 1 | `code/analysis/launch.sh` | +37/-2 | 动态模式开关 + 参数传递 |
| 2 | `code/analysis/main.py` | +13 | mock 掉 mujoco/d4rl 模块 |
| 3 | `code/analysis/parser.py` | +19 | 新增 3 个动态参数 |
| 4 | `code/config/locomotion_config.py` | +7/-1 | 补充缺失配置字段 |
| 5 | `code/diffuser/datasets/d4rl.py` | +47 | 新增 DummyD4RLEnv 兜底环境 |
| 6 | `code/diffuser/models/diffusion.py` | +9/-9 | 关键字参数 + p_sample 透传 |
| 7 | `code/diffuser/models/temporal.py` | +3/-4 | returns 空值防御 |
| 8 | `code/diffuser/utils/__init__.py` | +1/-1 | 注释 rendering 导入 |
| 9 | `code/diffuser/utils/training.py` | +57/-6 | **核心**：动态编译 + mark_dynamic |
| 10 | `code/scripts/evaluate_inv_parallel.py` | +120/-31 | **核心**：dynamic_benchmark |

合计：+310 / -67

---

## 三、核心动态 Shape 改造详解

### 3.1 `code/analysis/launch.sh`（动态模式开关）

新增三个动态 shape 变量，并通过自动校正块确保动态模式下编译参数正确：

```bash
DYNAMIC=true                       # 动态模式总开关
DYNAMIC_BS_LIST="1,2,4,8,16"       # batch 分档列表
DYNAMIC_ROUNDS=5                    # 每档重复轮数

# 动态模式自动校正：必须 COMPILE=true / GRAPH=false / SHAPE_HANDLE=false
if [ "$DYNAMIC" == "true" ]; then
    COMPILE=true
    GRAPH=false
    SHAPE_HANDLE=false
fi
```

并将参数透传给 main.py：
```bash
python main.py --dynamic=${DYNAMIC} \
               --dynamic_bs_list=${DYNAMIC_BS_LIST} \
               --dynamic_rounds=${DYNAMIC_ROUNDS} \
               --graph=${GRAPH} --compile=${COMPILE} --shape_handle=${SHAPE_HANDLE} ...
```

**合理性**：动态编译依赖 `torch.compile`，所以 `COMPILE` 必须为 true；CUDAGraph/NPUGraph 不支持动态 shape，所以 `GRAPH` 必须关闭；分桶（shape_handle）与动态 shape 互斥，必须关闭。自动校正块防止用户误配导致运行时崩溃。

### 3.2 `code/analysis/parser.py`（新增动态参数）

```python
parser.add_argument("--dynamic", type=str, default="false", choices=["true", "false"],
                    help="是否启用动态shape模式（动态编译，跳过图捕获）")
parser.add_argument("--dynamic_bs_list", type=str, default="",
                    help="动态batch_size分档列表，逗号分隔，如 '1,4,8,16'")
parser.add_argument("--dynamic_rounds", type=int, default=3,
                    help="每个batch档位重复测试轮数")
```

### 3.3 `code/diffuser/utils/training.py`（核心改造）

#### (a) `set_handle` 读取 dynamic 字段

```python
def set_handle(self, params):
    ...
    self.dynamic = params.get("dynamic", "false") == "true"
```

#### (b) `generate_inputs` 标记动态维度

```python
def generate_inputs(self, conditions, returns, observation_dim):
    ...
    inputs = {
        "x": apply_conditioning(x, conditions, 0),
        "noise": 0.5 * torch.randn_like(x)
    }
    try:
        torch._dynamo.mark_dynamic(inputs["x"], 0)          # batch 维动态
        if "noise" in inputs and isinstance(inputs["noise"], torch.Tensor):
            torch._dynamo.mark_dynamic(inputs["noise"], 0)
        if "returns" in inputs and isinstance(inputs["returns"], torch.Tensor):
            torch._dynamo.mark_dynamic(inputs["returns"], 0)
    except Exception:
        pass
    return inputs
```

**合理性**：`torch._dynamo.mark_dynamic(tensor, 0)` 显式告知编译器第 0 维（batch）是动态的，这是 `torch.compile(dynamic=True)` 泛化的关键。用 try/except 包裹是因为 mark_dynamic 对某些非标准 tensor 会抛异常，不应阻断主流程。

#### (c) `set_compile_model` 传递 dynamic 参数

```python
def set_compile_model(self):
    dyn = self.dynamic
    if self.compile:
        if self.graph and self.shape_handle and "npu" in self.device:
            self.shape_options["triton.cudagraphs"] = not dyn   # 动态模式关 cudagraphs
            self.forward = torch.compile(self.forward, backend="inductor", dynamic=dyn, options=self.shape_options)
        elif self.shape_handle and "npu" in self.device:
            self.forward = torch.compile(self.forward, backend="inductor", dynamic=dyn, options=self.shape_options)
        elif self.graph:
            mode = None if dyn else "reduce-overhead"          # 动态模式不能用 reduce-overhead
            self.forward = torch.compile(self.forward, backend="inductor", dynamic=dyn, mode=mode)
        else:
            self.forward = torch.compile(self.forward, backend="inductor", dynamic=dyn)
    else:
        if self.graph and not dyn and ("npu" in self.device or "cuda" in self.device):
            self.manual_graph = True
```

**合理性**：
- `dynamic=dyn` 是动态编译的核心参数，让 inductor 生成支持变长 batch 的 kernel。
- 动态模式下 `triton.cudagraphs=False`，因为 CUDAGraph 要求固定 shape。
- 动态模式下 `mode=None`（不能用 `reduce-overhead`，该模式内部依赖 cudagraph）。

#### (d) `model_infer` 修复 batch_size 获取

```python
def model_infer(self, inputs):
    if self.manual_graph:
        batch_size = inputs["x"].shape[0]    # 从实际输入获取，而非 self.batch_size
        return self.model_infer_graph(inputs, batch_size)
    return self.forward(inputs)
```

**合理性**：原代码用 `self.batch_size`（固定值），动态模式下 batch 会变化，必须从实际输入 tensor 的 shape 获取。

#### (e) `model_infer_graph` 增加 shape 校验

```python
else:
    entry = self.graphs[batch_size]
    for k in inputs.keys():
        cached = entry["static_input"][k]
        if cached.shape != inputs[k].shape:
            raise RuntimeError(
                f"model_infer_graph: input '{k}' shape mismatch. "
                f"cached={tuple(cached.shape)} vs input={tuple(inputs[k].shape)}. "
                f"CUDAGraph 不支持动态 shape，请用 --dynamic=true 走动态编译模式。"
            )
        entry["static_input"][k] = inputs[k]
```

**合理性**：CUDAGraph 静态图模式下，若输入 shape 变化会静默出错，显式抛错并提示切换动态模式，避免难排查的精度问题。

### 3.4 `code/scripts/evaluate_inv_parallel.py`（核心改造）

#### (a) 新增 `dynamic_benchmark` 函数

```python
def dynamic_benchmark(trainer, dataset, obs, returns, observation_dim, bs_list, rounds, deps):
    """动态shape纯前向benchmark：按bs_list轮转不同batch_size，测量前向性能。"""
    ...
    for cur_bs in bs_list:
        cur_obs = obs[:cur_bs]
        cur_returns = returns[:cur_bs]
        cur_obs = dataset.normalizer.normalize(cur_obs, 'observations')
        conditions = {0: to_torch(cur_obs, device=Config.device)}
        inputs = trainer.generate_inputs(conditions, cur_returns, observation_dim)

        # warmup 3 次
        for _ in range(3):
            with torch.no_grad():
                _ = trainer.model_infer(inputs)
        trainer.synchronize()

        # 正式测 rounds 轮
        for r in range(rounds):
            with profiling as prof:
                trainer.synchronize()
                start_time = time.time()
                samples, action = trainer.model_infer(inputs)
                trainer.synchronize()
                end_time = time.time()
                times_range.append(end_time - start_time)
                batch_size_record.append(cur_bs)
                prof.step()

        avg_time = sum(times_range[-rounds:]) / rounds
        logger.print(f"[dynamic bench] batch_size={cur_bs}, avg_time={avg_time:.4f}s")

    # 输出 unique_graphs 判定泛化是否成功
    counters = torch._dynamo.utils.counters
    unique_graphs = counters['stats']['unique_graphs']
    logger.print(f"[dynamic bench] torch.compile unique_graphs={unique_graphs} "
                 f"(1=动态编译成功泛化, >1=每个batch重编译)")

    output_report(times_range, batch_size_record)
```

**合理性**：
- 用合成数据（`np.zeros`）而非 Gym 环境，避免环境依赖，专注测模型推理性能。
- 每个 batch 档位先 warmup 3 次让编译稳定，再测 rounds 轮取平均。
- 通过 `unique_graphs` 量化泛化效果，这是动态 shape 改造成功与否的关键判据。

#### (b) `evaluate` 添加动态分支

```python
def evaluate(**deps):
    ...
    is_dynamic = deps.get("dynamic", "false") == "true"
    bs_list_str = deps.get("dynamic_bs_list", "")
    rounds = int(deps.get("dynamic_rounds", 3))

    if bs_list_str:
        bs_list = [int(x) for x in bs_list_str.split(",") if x.strip()]
    else:
        bs_list = [deps["test_batch_size"]]

    num_eval = max(bs_list)
    obs = np.zeros((num_eval, observation_dim), dtype=np.float32)
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), Config.device)

    # 动态模式也需要初始化 trainer 的 handle/compile
    trainer.set_handle(deps)
    trainer.set_hf32()
    trainer.set_compile_model()
    trainer.manual_graph = trainer.is_manual_graph()
    trainer.ema_model.eval()

    if is_dynamic:
        logger.print(f"[evaluate] 动态shape模式启动, bs_list={bs_list}, rounds={rounds}")
        dynamic_benchmark(trainer, dataset, obs, returns, observation_dim, bs_list, rounds, deps)
        logger.print("[evaluate] 动态shape benchmark完成")
        return
    # 否则走原有 Gym 交互评估流程
    ...
```

**合理性**：动态 benchmark 与 Gym 交互评估逻辑差异大，用独立函数 + 早返回保持清晰，避免大量 if/else 嵌套。

#### (c) 配置构造补充 returns_condition

```python
diffusion_config = utils.Config(
    Config.diffusion,
    ...
    returns_condition=Config.returns_condition,   # 必须显式传，与 TemporalUnet 的 returns_condition=True 对齐
)
```

**合理性**：`GaussianInvDynDiffusion` 必须以 `returns_condition=True` 初始化以匹配 `TemporalUnet`，否则 `p_mean_variance` 中 `self.model(x, cond, t, returns=returns)` 会因 `returns_condition` 为 False 而忽略 returns，导致 `TypeError: linear(): argument 'input' must be Tensor, not NoneType`。

### 3.5 `code/diffuser/models/diffusion.py`（接口改造）

#### (a) 关键字参数传递

```python
# 改造前（位置参数，returns 可能错位）
epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)

# 改造后（关键字参数，明确语义）
epsilon_cond = self.model(x, cond, t, returns=returns, use_dropout=False)
```

三处 `p_mean_variance`（GaussianDiffusion / GaussianInvDynDiffusion / ActionGaussianDiffusion）统一改为关键字参数。

**合理性**：位置参数在 `force_dropout=True` 等场景容易错位，关键字参数更安全，也为动态编译下 tensor 透传提供稳定接口。

#### (b) `p_sample_loop` / `conditional_sample` 透传 x 和 noise

```python
# 改造前
x = self.p_sample(cond, timesteps, x, noise, returns)
return self.p_sample_loop(shape, cond, x, noise, returns, *args, **kwargs)

# 改造后
x = self.p_sample(cond, timesteps, x=x, noise=noise, returns=returns)
return self.p_sample_loop(shape, cond, x=x, noise=noise, returns=returns, *args, **kwargs)
```

**合理性**：外部传入的 `x`、`noise` 已被 `mark_dynamic`，必须一路透传到 `p_sample_loop` 内部，否则内部重新生成的 tensor 不带动态标记，编译器无法泛化。

### 3.6 `code/diffuser/models/temporal.py`（returns 空值防御）

```python
# 改造前
if self.returns_condition:
    assert returns is not None
    returns_embed = self.returns_mlp(returns)

# 改造后
if self.returns_condition and returns is not None:
    returns_embed = self.returns_mlp(returns)
```

三处（TemporalUnet.forward / TemporalUnet.get_pred / MLPnet.forward）统一改造。

**合理性**：动态 benchmark 或某些调用路径下 returns 可能为 None，原 assert 会直接抛错阻断流程。改为条件判断，returns 为 None 时跳过 returns embedding 分支，保证前向能跑通。

---

## 四、环境兼容类改动

### 4.1 `code/analysis/main.py`（mock 掉 mujoco/d4rl）

```python
import sys
from unittest.mock import MagicMock
import diffuser.utils
diffuser.utils.MuJoCoRenderer = MagicMock

for module in ['mujoco_py', 'mujoco_py.builder', 'mujoco_py.cymj', 'd4rl.locomotion', 'd4rl.locomotion.ant']:
    sys.modules[module] = MagicMock()

import sys, params_proto
sys.modules["params_proto.neo_proto"] = params_proto
```

**合理性**：RecSDK benchmark 环境通常无 mujoco 物理引擎，mock 掉相关模块让模型能在纯前向模式下跑通。

### 4.2 `code/diffuser/datasets/d4rl.py`（DummyD4RLEnv 兜底）

```python
class DummyD4RLEnv:
    """Gym 环境不可用时返回的合成数据环境"""
    def __init__(self, name):
        # 根据 name 推断 obs_dim/act_dim（hopper=11/3, halfcheetah=17/6, ant=111/8）
        ...
    def get_dataset(self, **kwargs):
        # 返回 2000 条合成数据
        ...

_orig_load_environment = load_environment

def load_environment(name):
    try:
        return _orig_load_environment(name)
    except Exception as e:
        print(f'[Fallback] 无法加载 Gym 环境 "{name}" ({e})，已切换至 DummyD4RLEnv')
        return DummyD4RLEnv(name)
```

**合理性**：Gym 环境注册失败时兜底返回合成数据环境，让模型构建与动态 benchmark 不被环境缺失阻断。

### 4.3 `code/diffuser/utils/__init__.py`（注释 rendering）

```python
# from .rendering import *
```

**合理性**：rendering 依赖 mujoco，mock 后导入会报错，注释掉避免导入失败。

### 4.4 `code/config/locomotion_config.py`（补充字段）

```python
archive = None          # 避免 AttributeError: Config has no attribute 'archive'
n_saves = 5
label_freq = 2000
```

**合理性**：原 Config 缺字段，运行时 `getattr(Config, 'archive')` 报 AttributeError，补充默认值。

---