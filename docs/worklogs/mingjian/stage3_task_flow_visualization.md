# 阶段 3 CALVIN 任务流程可视化

## 当前完整流程

```mermaid
flowchart TD
    A["① 接收 CALVIN 官方任务<br/><b>push the sliding door to the left side</b>"]
    B["② 拍摄当前画面<br/>静态相机 + 夹爪相机 + proprio"]
    C["③ Qwen Planner 提出计划<br/>原始建议保留用于训练和诊断"]
    D{"④ CALVIN backend contract<br/>检查执行契约"}
    E["⑤ 生成唯一原子子目标 S1<br/><b>官方原句逐字直通</b><br/>候选框与 candidate id 可选"]
    F["⑥ Preflight 安全检查<br/>图像、机器人状态、环境、HTTP backend"]
    G["⑦ X-VLA HTTP 推理<br/>输入：当前图像 + proprio + 官方原句"]
    H["⑧ 执行动作块<br/>每块最多 20 个环境步"]
    I{"⑨ CALVIN task oracle<br/>完整任务成功了吗？"}
    J["⑩ 立即收车<br/>loop = finished<br/>reason = environment_oracle_success<br/>plan/subgoal = succeeded"]
    K["⑩a 拍摄 post-action 验证画面"]
    L{"⑪ Visual Verifier<br/>判断进度与下一步"}
    M["继续执行<br/>回到 Preflight，再取一个动作块"]
    N["重新观察 / 重新规划 / 恢复<br/>按失败类型选择修复路线"]
    X["环境或 HTTP 异常<br/>明确记录失败，不冒充任务失败或成功"]

    A --> B --> C --> D
    D -->|"强制：1 个子目标"| E
    E --> F --> G --> H --> I
    I -->|"success = true<br/>最高优先级"| J
    I -->|"success = false"| K --> L
    L -->|"continue_execute"| M --> F
    L -->|"reobserve / replan / recover"| N
    N --> B
    F -. "检查失败" .-> X
    G -. "超时、非 2xx、坏动作" .-> X
    H -. "环境 step 异常" .-> X

    classDef input fill:#dbeafe,stroke:#2563eb,color:#172554,stroke-width:2px;
    classDef planning fill:#ede9fe,stroke:#7c3aed,color:#2e1065,stroke-width:2px;
    classDef execution fill:#fef3c7,stroke:#d97706,color:#451a03,stroke-width:2px;
    classDef oracle fill:#dcfce7,stroke:#16a34a,color:#052e16,stroke-width:3px;
    classDef finish fill:#bbf7d0,stroke:#15803d,color:#052e16,stroke-width:3px;
    classDef recovery fill:#ffedd5,stroke:#ea580c,color:#431407,stroke-width:2px;
    classDef error fill:#fee2e2,stroke:#dc2626,color:#450a0a,stroke-width:2px;

    class A,B input;
    class C,D,E planning;
    class F,G,H execution;
    class I oracle;
    class J finish;
    class K,L,M,N recovery;
    class X error;
```

## 旧流程与新流程对比

```mermaid
flowchart LR
    subgraph OLD["旧流程：机械臂到了终点，调度员还让它绕圈"]
        O1["Planner 拆成<br/>抓取 → 移动 → 释放"] --> O2["X-VLA 收到局部指令<br/>偏离官方训练语言"]
        O2 --> O3["Oracle 已成功"]
        O3 --> O4["仍进入 Verifier"]
        O4 --> O5["动作后重新定位"]
        O5 --> O6["连续空候选"]
        O6 --> O7["stalled_loop"]
    end

    subgraph NEW["新流程：终点绿旗一举，立即收车"]
        N1["Planner 原始建议"] --> N2["Backend 收束为<br/>1 条官方原句"]
        N2 --> N3["X-VLA 执行"]
        N3 --> N4{"Oracle 成功？"}
        N4 -->|"是"| N5["立即 finished"]
        N4 -->|"否"| N6["Verifier 判断进度"]
        N6 --> N3
    end

    classDef bad fill:#fee2e2,stroke:#dc2626,color:#450a0a,stroke-width:2px;
    classDef good fill:#dcfce7,stroke:#16a34a,color:#052e16,stroke-width:2px;
    class O1,O2,O3,O4,O5,O6,O7 bad;
    class N1,N2,N3,N4,N5,N6 good;
```

## 一句话理解各角色

|角色|形象比喻|职责|
|---|---|---|
|Planner|接单员|理解任务并提出计划；原始建议保留供训练|
|CALVIN backend contract|工单闸机|把计划收束成 X‑VLA 真正会执行的一条官方原子指令|
|Preflight|发车检查员|检查相机、机器人状态、环境和动作服务是否健康|
|X‑VLA|机械臂驾驶员|根据图像、proprio 和官方指令输出动作块|
|Verifier|途中观察员|oracle 尚未成功时，判断继续、重观察还是恢复|
|CALVIN oracle|终点裁判|对完整任务拥有最高判定权；绿旗一举，Agent 立即结束|

## 本次真实成功轨迹

```text
接单 → 拍照 → 原子计划 → Preflight
     → 20 步动作 → 未成功 → Verifier → 继续
     → 20 步动作 → 未成功 → Verifier → 继续
     → 20 步动作 → 未成功 → Verifier → 继续
     → 第 61 个环境步 → Oracle 成功 → 当场结束
```

关键数字：61 个环境步、37 个 Agent 决策、0 次 localization、0 个失败技能。
