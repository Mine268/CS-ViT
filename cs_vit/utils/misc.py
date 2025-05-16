from typing import *
import warnings
from functools import partial
from colorama import Fore, Style, init
init(autoreset=True)

import torch
import numpy as np


def breif_dict(output: dict, prefix=""):
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"{prefix}{k}: tensor, {list(v.shape)}")
        elif isinstance(v, np.ndarray):
            print(f"{prefix}{k}: array, {list(v.shape)}")
        elif isinstance(v, (str, int, float, list, tuple)):
            print(f"{prefix}{k}: {type(v).__name__}, {v}")
        elif v is None:
            print(f"{prefix}{k}: None")
        else:
            breif_dict(v, f"{k}.")


def to_tuple(x: Any | tuple) -> tuple:
    if isinstance(x, tuple):
        return x
    else:
        return (x, x)


def move_to_device(
    data: Union[Dict[str, Any], List[Any], torch.Tensor], device: torch.device
) -> Union[Dict[str, Any], List[Any], torch.Tensor]:
    if isinstance(data, dict):
        for k in data:
            move_to_device(data[k], device)
    elif isinstance(data, list):
        for i in range(len(data)):
            move_to_device(data[i], device)
    elif isinstance(data, torch.Tensor):
        data.data = data.to(device, non_blocking=False)
    return data


def flatten_dict(d, parent_key="", sep="/"):
    for k, v in d.items():
        current_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            yield from flatten_dict(v, current_key, sep)
        else:
            yield current_key, v


def get_tensor_memory(tensor: torch.Tensor) -> int:
    """计算单个Tensor的内存占用（包括存储和梯度）"""
    # 屏蔽弃用警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if tensor.is_cuda:
            # GPU Tensor直接计算元素总数×元素大小
            return tensor.element_size() * tensor.nelement()
        else:
            # CPU Tensor使用untyped_storage替代storage()
            if hasattr(tensor, 'untyped_storage'):
                return tensor.untyped_storage().size() * tensor.element_size()
            else:
                # 兼容旧版PyTorch
                return tensor.storage().size() * tensor.element_size()


def stat_dict_memory(data_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, str]:
    """优化后的内存统计函数"""
    result = {}

    for key, value in data_dict.items():
        try:
            if isinstance(value, torch.Tensor):
                # 处理单个Tensor
                mem_bytes = get_tensor_memory(value)
                grad_mem = get_tensor_memory(value.grad) if value.grad is not None else 0
                total = mem_bytes + grad_mem
                device = 'GPU' if value.is_cuda else 'CPU'
                result[key] = f"{total / 1024**2:.2f} MB ({device} Tensor) + grad {grad_mem / 1024**2:.2f} MB"

            elif isinstance(value, list) and all(isinstance(x, torch.Tensor) for x in value):
                # 处理List[Tensor]
                total = sum(get_tensor_memory(tensor) for tensor in value)
                grad_total = sum(get_tensor_memory(tensor.grad) for tensor in value if tensor.grad is not None)
                device = 'GPU' if value[0].is_cuda else 'CPU' if len(value) > 0 else 'N/A'
                result[key] = f"{total / 1024**2:.2f} MB (List[{len(value)} {device} Tensors]) + grads {grad_total / 1024**2:.2f} MB"

            else:
                raise TypeError(f"Unsupported type for key '{key}': {type(value)}")

        except Exception as e:
            result[key] = f"Error: {str(e)}"

    return result


def print_with_prefix(*args, prefix="> ", **kwargs):
    """
    包装print函数，在每一行输出前添加自定义前缀

    Args:
        *args: 要打印的参数，与原print函数一致
        prefix: 要添加的前缀，默认为"> "
        **kwargs: 其他传递给print的关键字参数
    """
    # 将多个参数转换为单个字符串，用空格分隔（与原print一致）
    message = ' '.join(str(arg) for arg in args)

    # 分割成多行
    lines = message.splitlines()

    # 如果没有内容，直接调用原始print（处理print()的情况）
    if not lines:
        print(**kwargs)
        return

    # 为每一行添加前缀
    prefixed_lines = [prefix + line for line in lines]

    # 重新组合成完整消息
    prefixed_message = '\n'.join(prefixed_lines)

    # 调用原始print函数
    print(prefixed_message, **kwargs)


def wrap_prefix_print(prefix: str):
    return partial(print_with_prefix, prefix=prefix)


def print_grouped_losses(epoch, iteration, total_iters, iter_time, lr, forward_result, print_):
    """
    通用分组损失日志输出

    Args:
        epoch: 当前epoch
        iteration: 当前迭代次数 (0-based)
        total_iters: 总迭代次数
        iter_time: 每次迭代时间(秒)
        lr: 当前学习率
        forward_result: 包含losses的字典，格式如下:
            {
                "loss": total_loss,
                "logs": {
                    "scalar": {
                        "total": total_loss,
                        "group1": {
                            "group1": main_loss_for_group1,  # 与组名相同->红色
                            "component1": sub_loss1,         # 其他->蓝色
                            "component2": sub_loss2
                        },
                        "group2": {
                            "group2": main_loss_for_group2,
                            "componentA": sub_lossA
                        }
                    }
                }
            }
    """
    # 头部信息 (蓝色加粗)
    header = (
        Fore.BLUE + Style.BRIGHT +
        f"Epoch {epoch} [{iteration + 1}/{total_iters}]" +
        Style.RESET_ALL
    )

    # 时间信息 (黄色)
    time_info = (
        Fore.YELLOW +
        f" | iter: {iter_time}" +
        f" | ETA: {iter_time * (total_iters - iteration - 1)}" +
        Style.RESET_ALL
    )

    # 学习率 (青色)
    lr_info = (
        Fore.CYAN +
        f" | lr: {lr:.4e}" +
        Style.RESET_ALL
    )

    # 总损失 (红色加粗)
    total_loss = (
        Fore.RED + Style.BRIGHT +
        f" | Total: {forward_result['logs']['scalar']['total']:.6f}" +
        Style.RESET_ALL
    )

    # 构建分组损失信息
    def format_group_losses(group_name, group_dict):
        lines = []
        main_loss = None

        # 先处理与组名相同的key (主损失)
        if group_name in group_dict:
            main_loss = (
                Fore.RED + Style.BRIGHT +
                f"{group_name}: {group_dict[group_name]:.6f}" +
                Style.RESET_ALL
            )
            lines.append(main_loss)

        # 处理其他损失组件
        components = []
        for k, v in group_dict.items():
            if k != group_name:  # 跳过已处理的主损失
                components.append(
                    Fore.BLUE + f"{k}: {v:.6f}" + Style.RESET_ALL
                )

        if components:
            lines.append("(" + ", ".join(components) + ")")

        return " ".join(lines)

    # 收集所有分组信息
    group_lines = []
    scalar_logs = forward_result['logs']['scalar']

    for group_name in scalar_logs:
        if isinstance(scalar_logs[group_name], dict):  # 只处理字典类型的组
            formatted = format_group_losses(group_name, scalar_logs[group_name])
            group_lines.append(formatted)

    # 组合所有信息
    log_str = (
        f"{header}{time_info}{lr_info}{total_loss}\n" +
        "\n".join(f"  * {line}" for line in group_lines)
    )

    print_(log_str)