# 铭健本机服务与端口地图

最后检查：2026-07-16。端口不是永久车位，每次启动前仍要重新确认占用者。

本机服务像一排有编号的实验台：即使插头能插进去，也必须先确认这张台子属于谁、接了哪张 GPU、正在跑什么模型。误杀一个陌生 PID，可能等于拔掉其他成员正在进行的长实验。

|服务|地址|GPU 或进程环境|当前状态|说明|
|---|---|---|---|---|
|X-VLA `/act`|`127.0.0.1:8000`|`xvla-stable`，1 张 A6000|已验证，测试后停止|阶段 2 真实服务加载了已校验 hash 的 checkpoint 并处理全部请求，随后释放端口|
|ClawVLA PolicyProxy|`127.0.0.1:18080`|未来的 OpenRLHF 进程|空闲，未运行|来自 RL policy 配置|
|OpenPI 默认 probe|`127.0.0.1:8765`|CALVIN 不使用|空闲，未运行|CALVIN 使用独立 X-VLA 服务|
|旧环境 worker 基址|`127.0.0.1:18765`|当前 CALVIN smoke 未启用|空闲，未运行|共享 RL 代码中的字段名仍保留 RoboTwin 历史命名|
|本机系统代理|`127.0.0.1:7890`|系统服务|已占用|不属于本项目，禁止停止|
|已有未知本机服务|`127.0.0.1:9000–9004`|其他进程或用户|已占用|不复用、不终止|

## localhost 必需环境变量

```bash
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
```

这两行相当于告诉请求“隔壁房间就在走廊对面，不要先绕到外网代理再回来”。缺少它们时，本机 PolicyProxy 测试曾收到 HTTP 502。

不要使用宽泛的 `pkill`，也不要终止身份不明的 listener。启动前要核对端口、PID、命令和所有者。健康检查还必须确认模型/checkpoint 身份，不能只看到 HTTP 200 就认定接对了服务。

## 已验证的 X-VLA 启动命令

```bash
cd /home2/gmj/X-VLA
conda activate xvla-stable
CUDA_VISIBLE_DEVICES=0 \
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
PYTHONPATH=/home2/gmj/X-VLA \
/home2/gmj/miniconda3/envs/xvla-stable/bin/python -m deploy \
  --model_path /home2/gmj/models/X-VLA-Calvin-ABC_D \
  --host 127.0.0.1 \
  --port 8000 \
  --output_dir /home2/gmj/Agent_skill/EmbodiedSkills/tmp_runs/xvla_calvin_server \
  --disable_slurm
```

上游服务没有单独的 health route。判断“真正就绪”需要同时看到 Uvicorn 启动日志和一条通过合约校验的真实 `/act` 请求。阶段 2 运行时服务约占 4.0 GiB GPU 显存；验收结束后已关闭，端口和显存均释放。
