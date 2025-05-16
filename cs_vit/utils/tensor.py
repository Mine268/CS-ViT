import torch


def calculate_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # 计算单个参数的L2范数
            total_norm += param_norm.item() ** 2  # 平方后累加
    total_norm = total_norm * 0.5  # 开平方得到总梯度模
    return total_norm
