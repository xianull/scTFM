import logging

def get_pylogger(name=__name__) -> logging.Logger:
    """
    获取 Logger 实例。
    
    在 Hydra 项目中，日志的格式（颜色、时间等）和输出位置（控制台、文件）
    应该完全由 `configs/hydra/default.yaml` 控制。
    
    这里不需要手动添加 Handler，否则会导致和 Hydra 的 Handler 冲突，
    出现双重日志 (Double Logging)。
    """
    return logging.getLogger(name)