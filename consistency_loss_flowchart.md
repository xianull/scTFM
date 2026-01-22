# Consistency Loss 实现流程图

```mermaid
flowchart TD
    Start([输入: x_seq, time_seq, seq_len]) --> Extract[提取序列中的三个关键点]
    
    Extract --> StartPoint["x_start, t_start<br/>(序列起点)"]
    Extract --> MidPoint["x_mid, t_mid<br/>(序列中点, mid_idx = slen // 2)"]
    Extract --> EndPoint["x_end, t_end<br/>(序列终点)"]
    
    StartPoint --> CalcDT[计算时间差]
    MidPoint --> CalcDT
    EndPoint --> CalcDT
    
    CalcDT --> DT1["dt1 = t_mid - t_start"]
    CalcDT --> DT2["dt2 = t_end - t_mid"]
    CalcDT --> DTTotal["dt_total = t_end - t_start"]
    
    DT1 --> DirectPath[直接路径: start → end]
    DT2 --> DirectPath
    DTTotal --> DirectPath
    
    DirectPath --> ZDirect["z_direct = 0.5 * (x_start + x_end)<br/>插值中间点"]
    ZDirect --> VDirect["v_direct = backbone(z_direct, t=0.5, cond_direct)<br/>直接路径速度场"]
    
    DT1 --> Path1[路径1: start → mid]
    Path1 --> Z1["z1 = 0.5 * (x_start + x_mid)<br/>插值中间点"]
    Z1 --> V1["v1 = backbone(z1, t=0.5, cond1)<br/>第一段速度场"]
    
    DT2 --> Path2[路径2: mid → end]
    Path2 --> Z2["z2 = 0.5 * (x_mid + x_end)<br/>插值中间点"]
    Z2 --> V2["v2 = backbone(z2, t=0.5, cond2)<br/>第二段速度场"]
    
    V1 --> Compose[组合速度场]
    V2 --> Compose
    DT1 --> Compose
    DT2 --> Compose
    
    Compose --> VComposed["v_composed = (v1 * dt1 + v2 * dt2) / (dt1 + dt2)<br/>加权平均组合速度场"]
    
    VDirect --> Loss[MSE Loss]
    VComposed --> Loss
    
    Loss --> ConsistencyLoss["loss_cons = MSE(v_direct, v_composed)<br/>速度场一致性损失"]
    
    ConsistencyLoss --> End([返回 loss_cons])
    
    style DirectPath fill:#e1f5ff
    style Path1 fill:#fff4e1
    style Path2 fill:#fff4e1
    style Compose fill:#e8f5e9
    style Loss fill:#ffebee
    style ConsistencyLoss fill:#ffebee
```

## 核心思想

1. **速度场一致性约束**：直接跨越路径 (start → end) 的速度场预测应该与分步跨越路径 (start → mid → end) 的组合速度场一致

2. **关键步骤**：
   - 在每条路径的中间点（flow time t=0.5）查询速度场
   - 直接路径：`z_direct = 0.5 * (x_start + x_end)`，查询 `v_direct`
   - 分步路径：分别查询 `v1` (start→mid) 和 `v2` (mid→end)
   - 组合速度场：`v_composed = (v1 * dt1 + v2 * dt2) / (dt1 + dt2)`（按时间比例加权）

3. **优势**：
   - 避免多步采样，直接比较速度场
   - 有梯度，可高效训练
   - 每 N 步计算一次（`cons_every_n_steps`）以加速训练

4. **Loss 公式**：
   ```
   loss_cons = MSE(v_direct, v_composed)
   ```
