import torch
import time
from flashinfer.sampling import top_k_top_p_sampling_from_probs
from flashinfer.sampling import top_k_top_p_filter_return_probs
import numpy as np
"""
export CUDA_VISIBLE_DEVICES=7
python -m pip install --no-build-isolation -e . -v
"""
def golden_impl(prob, topk, topp):
    batch_size, vocab_size = prob.shape
    device = prob.device
    topk = topk.long().clamp(min=1, max=vocab_size)          # 确保至少保留一个词
    topp = topp.float()

    # 为了高效，一次性取所有样本的前 max_k 个最大值
    max_k = topk.max().item()
    values, indices = torch.topk(prob, max_k, dim=-1)         # (batch, max_k)

    # 初始化输出
    filtered = torch.zeros_like(prob)

    for i in range(batch_size):
        k_i = topk[i].item()
        p_i = topp[i].item()

        # 当前样本的前 k_i 个概率值（已降序）
        vals_i = values[i, :k_i]           # (k_i,)
        cumsum = torch.cumsum(vals_i, dim=0)   # 累积和

        # 找到满足累积和 < p_i 的最大索引（至少保留一个）
        mask = cumsum < p_i
        if mask.any():
            m = mask.sum().item()           # 前 m 个满足条件
        else:
            m = 1                           # 第一个词概率已经 > p_i，但仍保留它

        # 保留的索引
        keep_indices = indices[i, :m]       # (m,)
        # 将原概率值赋给输出
        filtered[i, keep_indices] = prob[i, keep_indices]
    return filtered

def generate_llm_like_prob(
    batch_size: int,
    vocab_size: int,
    distribution: str = "power_law",
    temperature: float = 1.0,
    spike_ratio: float = 0.1,        # spike 分布中高概率 token 的比例
    spike_mass: float = 0.8,          # spike 分布中高概率 token 占的总质量
    device: str = "cuda"
) -> torch.Tensor:
    """
    生成类似 LLM 输出的概率分布张量。

    Args:
        batch_size: 批次大小
        vocab_size: 词表大小
        distribution: 分布类型，可选 "power_law", "spike", "temperature"
        temperature: 温度参数（仅对 temperature 分布有效），<1 使分布更尖锐
        spike_ratio: spike 分布中高概率 token 的比例（例如 0.1 表示前 10% token 占据大部分概率）
        spike_mass: spike 分布中高概率 token 占据的总概率质量（例如 0.8 表示前 10% token 共占 80% 概率）
        device: 设备，默认为 "cuda"

    Returns:
        torch.Tensor: 形状 (batch_size, vocab_size)，每行和为 1
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = "cpu"

    if distribution == "power_law":
        # 幂律分布：概率与排名成反比（z = 1.0 的 Zipf 分布）
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
        # 概率 ~ 1/rank，归一化后得到长尾分布
        prob = 1.0 / ranks
        prob = prob / prob.sum()
        # 扩展到 batch_size
        prob = prob.unsqueeze(0).expand(batch_size, -1)
        return prob

    elif distribution == "spike":
        # 尖峰分布：前 spike_ratio 部分 token 共享 spike_mass 总质量，其余均匀
        prob = torch.zeros(batch_size, vocab_size, device=device)
        n_spike = max(1, int(vocab_size * spike_ratio))
        # 为每个 batch 独立生成高概率 token 的索引（可重复，但通常我们希望每个 batch 的尖峰位置不同）
        # 这里采用：所有 batch 共享相同的尖峰位置（前 n_spike 个 token）
        # 如果想要每个 batch 不同，可随机采样，但为了简单先共享
        # 高概率部分分配总质量 spike_mass，并按指数衰减分配（也可均匀）
        # 为了模拟更真实，让高概率部分内部也遵循衰减
        spike_weights = torch.exp(-torch.arange(n_spike, dtype=torch.float32, device=device) / (n_spike / 3))
        spike_weights = spike_weights / spike_weights.sum() * spike_mass
        # 低概率部分分配剩余质量，均匀分配
        low_mass = 1.0 - spike_mass
        low_prob = low_mass / (vocab_size - n_spike) if vocab_size > n_spike else 0.0
        prob[:, :n_spike] = spike_weights
        if vocab_size > n_spike:
            prob[:, n_spike:] = low_prob
        # 确保每行严格和为1（浮点误差可能略偏差）
        prob = prob / prob.sum(dim=-1, keepdim=True)
        return prob

    elif distribution == "temperature":
        # 从正态分布生成 logits，通过温度控制分布尖锐程度
        logits = torch.randn(batch_size, vocab_size, device=device)
        # 温度调整：温度越低，softmax 越尖锐
        logits = logits / temperature
        prob = torch.softmax(logits, dim=-1)
        return prob
    elif distribution == "random":
        return torch.softmax(torch.randn(batch_size, vocab_size, device=device), dim=-1)
    else:
        raise ValueError("distribution 必须是 'power_law', 'spike' 或 'temperature'")

def test_case(func_name, func, num_warmup, num_runs, *args, **kwargs):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    for i in range(num_warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    times_func = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        times_func.append(end_time - start_time)
    
    # 计算统计信息
    avg_time_func = sum(times_func) / num_runs * 1000  # 转换为毫秒
    min_time_func = min(times_func) * 1000
    max_time_func = max(times_func) * 1000
    return (func_name, avg_time_func, ret)

def test_top_k_top_p_sampling_performance(batch_size=16):
    """测试top-k和top-p采样函数的性能，batch size=64，所有batch使用相同的top-k和top-p"""
    print(50*"#")

    # 测试配置
    vocab_size = 151936  # LLM词表大小
    num_runs = 400  # 运行多次取平均
    top_k_value = 1024
    top_p_value = 0.98

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备测试数据 - 放在cuda上
    # 使用尖峰分布
    probs = generate_llm_like_prob(batch_size, vocab_size, distribution="spike", spike_ratio=0.2, spike_mass=0.7, device=device)
    # 低温分布（尖锐）
    # probs = generate_llm_like_prob(batch_size, vocab_size, distribution="temperature", temperature=0.5, device=device)
    # 创建相同的top-k和top-p tensor（所有batch相同）
    top_ks = torch.full((batch_size,), top_k_value, dtype=torch.long, device=device)
    top_ps = torch.full((batch_size,), top_p_value, dtype=torch.float32, device=device)

    case1_ret = test_case(
        "top_k_top_p_filter_return_probs", 
        top_k_top_p_filter_return_probs, 
        10,
        400,
        probs.contiguous(),
        top_ks,
        top_ps,
        filter_apply_order="joint",
        check_nan=False,
    )

    case2_ret = test_case(
        "top_k_top_p_sampling_from_probs", 
        top_k_top_p_sampling_from_probs, 
        10,
        400,
        probs.contiguous(),
        top_ks,
        top_ps,
        filter_apply_order="joint",
        check_nan=False,
    )
    
    # 打印性能结果
    print(f"性能对比 (函数2 vs 函数1): {case1_ret[1]/case2_ret[1]:.2f}x")
    print(f"(batch_size={batch_size}, vocab_size={vocab_size}, top_k={top_k_value}, top_p={top_p_value}")
    info_str = (f"函数1 - {case1_ret[0]}: 平均时间: {case1_ret[1]:.3f} ms")
    print(info_str)
    info_str = (f"函数2 - {case2_ret[0]}: 平均时间: {case2_ret[1]:.3f} ms")
    print(info_str)

if __name__ == "__main__":
    test_top_k_top_p_sampling_performance(1)
    test_top_k_top_p_sampling_performance(16)
    test_top_k_top_p_sampling_performance(32)
    test_top_k_top_p_sampling_performance(64)
    test_top_k_top_p_sampling_performance(128)
    test_top_k_top_p_sampling_performance(256)
