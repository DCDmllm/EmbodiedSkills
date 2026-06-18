# Legacy VERL RL Backend

当前 ClawVLA/RoboTwin RL 主训练路径是 OpenRLHF：

```bash
PYTHONPATH=src .venv-openrlhf-py310-cu128/bin/python -m clawvla.rl.openrlhf_runner \
  --config configs/rl/qwen3vl_pi05_multitask_1update.yaml \
  --mode dry-run
```

旧 VERL 实现已经归档到：

```text
src/clawvla/rl/legacy_verl/runner.py
src/clawvla/rl/legacy_verl/verl_agent_loop.py
src/clawvla/rl/legacy_verl/verl_runtime_patches.py
```

为避免旧 import 立即失效，原路径保留了兼容 wrapper：

```text
src/clawvla/rl/runner.py
src/clawvla/rl/verl_agent_loop.py
src/clawvla/rl/verl_runtime_patches.py
```

旧 VERL 命令仍可通过 legacy 脚本运行：

```bash
./scripts/run_clawvla_verl_legacy.sh --config configs/rl/qwen3vl_pi05_grpo.yaml --mode dry-run
```

不要再基于 VERL 路径开发新训练功能。新的 trajectory agent RL、真实 RoboTwin rollout、多任务配置和多卡训练入口都应走 OpenRLHF。
