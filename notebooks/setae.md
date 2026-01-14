
---

### 方案一：空间图自编码器 
通过图卷积（GCN）或图注意力（GAT）在编码阶段就强制引入微环境信息。
- **输入设计：** 将单细胞数据表示为一个图 $G = (V, E)$，其中 $V$ 是细胞节点的基因表达特征，$E$ 是根据物理坐标构建的 $K$-NN 邻接矩阵。
- **编码器 (Encoder)：**
    - 使用 2-3 层 **GAT (Graph Attention Network)**。
    - **关键点：** GAT 的注意力机制可以学习到哪些邻居对中心细胞的影响更大，从而更好地建模微环境。
- **双重解码器 (Dual Decoder)：**
    1. **特征重建层：** 重建中心细胞的基因表达向量 $\hat{X}$。
    2. **结构重建层（链路预测）：** 预测细胞间的连接概率 $\hat{A}$。这能强迫 Latent 空间保留x-space的拓扑结构

#### 架构图：Spatial Graph AE (One-Stage)

```mermaid
flowchart LR
    subgraph Input["输入层"]
        X["基因表达矩阵 X<br/>(N × G)"]
        A["K-NN 邻接矩阵 A<br/>(基于空间坐标)"]
    end
    
    subgraph GATEncoder["GAT 编码器"]
        X --> GAT1["GAT Layer 1"]
        A --> GAT1
        GAT1 --> H1["H1"]
        H1 --> GAT2["GAT Layer 2"]
        A --> GAT2
        GAT2 --> H2["H2"]
        H2 --> GAT3["GAT Layer 3"]
        A --> GAT3
        GAT3 --> Z["Latent Z<br/>(融合邻居信息)"]
    end
    
    subgraph DualDecoder["双重解码器"]
        Z --> FeatDec["特征重建<br/>Decoder"]
        FeatDec --> Xhat["X̂<br/>(重建表达)"]
        
        Z --> LinkPred["链路预测<br/>(Z @ Z.T)"]
        LinkPred --> Ahat["Â<br/>(重建邻接)"]
    end
    
    subgraph Loss["损失函数"]
        Xhat --> Lrec["L_rec<br/>(MSE/NB)"]
        Ahat --> Llink["L_link<br/>(BCE)"]
        Lrec --> Total["L = L_rec + λ·L_link"]
        Llink --> Total
    end
    
    style Input fill:#e3f2fd
    style GATEncoder fill:#fff3e0
    style DualDecoder fill:#f3e5f5
    style Loss fill:#e8f5e9
```

---

### 方案二：Contrastive AE
，强制 Latent 空间在局部邻域内保持平滑。
- **核心思想：** 协同优化重建损失和邻域对比损失。
- 损失函数设计：$$L = L_{rec} + \lambda L_{contrastive}$$
- **$L_{contrastive}$ 的实现：**
    - **正样本对：** x-space中的 $K$ 个邻居细胞。
    - **负样本对：** 随机采样的非邻居细胞。
    - 使用 **InfoNCE Loss**，强制中心细胞的 Latent 表示 $z_i$ 与其物理邻居 $z_j$ 的距离尽可能近。这会直接修正你之前图中红色点（z-space 邻居）和蓝色点（x-space 邻居）不统一的问题。

#### 架构图：Contrastive AE (One-Stage)

```mermaid
flowchart TB
    subgraph Input["输入采样"]
        Xi["中心细胞 x_i"]
        Pos["正样本<br/>(K个空间邻居)"]
        Neg["负样本<br/>(随机非邻居)"]
    end
    
    subgraph Encoder["共享 Encoder"]
        Xi --> Enc["Encoder<br/>(MLP/Transformer)"]
        Pos --> Enc
        Neg --> Enc
        Enc --> Zi["z_i"]
        Enc --> Zpos["z_pos"]
        Enc --> Zneg["z_neg"]
    end
    
    subgraph Decoder["Decoder"]
        Zi --> Dec["Decoder"]
        Dec --> Xhat["重建 x̂_i"]
    end
    
    subgraph ContrastiveLoss["对比学习模块"]
        Zi --> Sim1["相似度计算"]
        Zpos --> Sim1
        Sim1 --> PosScore["正样本得分<br/>(拉近)"]
        
        Zi --> Sim2["相似度计算"]
        Zneg --> Sim2
        Sim2 --> NegScore["负样本得分<br/>(推远)"]
        
        PosScore --> InfoNCE["InfoNCE Loss"]
        NegScore --> InfoNCE
    end
    
    subgraph TotalLoss["总损失"]
        Xhat --> Lrec["L_rec"]
        InfoNCE --> Lcon["L_contrastive"]
        Lrec --> Total["L = L_rec + λ·L_contrastive"]
        Lcon --> Total
    end
    
    style Input fill:#e3f2fd
    style Encoder fill:#fff3e0
    style Decoder fill:#f3e5f5
    style ContrastiveLoss fill:#ffebee
    style TotalLoss fill:#e8f5e9
```

---

### 方案三：掩码自编码器 Context-Aware Masked AE
借鉴 **scGPT** /**MAE**/**cellBERT** 的思路，通过"预测缺失信息"来理解微环境。
- **设计逻辑：**
    1. 输入一个中心细胞和它周围的 $N$ 个邻居细胞作为一个输入 Patch。
    2. **掩码：** 随机遮盖中心细胞的部分高变基因，或者直接遮盖整个中心细胞的表达。
    3. **预测：** 让编码器利用邻居细胞的特征来预测（补全）中心细胞被遮盖的信息。
- **优点：** 这种方案能迫使模型学习到"微环境如何决定细胞状态"的条件概率，而不仅仅是简单的特征降维。

#### 架构图：Context-Aware Masked AE (One-Stage)

```mermaid
flowchart TB
    subgraph Input["输入 Patch"]
        Center["中心细胞 x_i"]
        N1["邻居 x_1"]
        N2["邻居 x_2"]
        Ndot["..."]
        Nn["邻居 x_N"]
    end
    
    subgraph Masking["掩码策略"]
        Center --> Mask["随机掩码<br/>(部分基因 or 整个细胞)"]
        Mask --> MaskedCenter["[MASK] 表示"]
    end
    
    subgraph ContextEncoder["上下文编码器"]
        MaskedCenter --> TokenEmb["Token Embedding"]
        N1 --> TokenEmb
        N2 --> TokenEmb
        Ndot --> TokenEmb
        Nn --> TokenEmb
        
        TokenEmb --> Trans["Transformer<br/>Encoder"]
        Trans --> ContextZ["上下文感知<br/>表征 z_context"]
    end
    
    subgraph Prediction["预测头"]
        ContextZ --> PredHead["Prediction Head<br/>(MLP)"]
        PredHead --> PredX["预测被掩码的<br/>基因表达"]
    end
    
    subgraph Loss["损失函数"]
        PredX --> Compare["比较"]
        Center --> GT["Ground Truth"]
        GT --> Compare
        Compare --> Lmask["L_mask<br/>(仅计算被掩码部分)"]
    end
    
    style Input fill:#e3f2fd
    style Masking fill:#ffcdd2
    style ContextEncoder fill:#fff3e0
    style Prediction fill:#f3e5f5
    style Loss fill:#e8f5e9
```
