import sys
import os

# 获取当前脚本所在的绝对路径 (即项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将根目录添加到 sys.path 中
# 这样 Python 解释器就能识别 'src' 为一个可导入的包
sys.path.append(current_dir)

# 从 src/train.py 中导入被 @hydra.main 装饰的 main 函数
from src.train import main

if __name__ == "__main__":
    # 启动训练
    main()