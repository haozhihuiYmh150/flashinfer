"""
export CUDA_VISIBLE_DEVICES=7
python -m pip install --no-build-isolation -e . -v
"""

import torch
import time
from flashinfer.sampling import top_k_top_p_sampling_from_probs
from flashinfer.sampling import top_k_top_p_filter_return_probs
import numpy as np

"""
# 第一次修改代码后，编译运行
python test_return_probs.py baseline

# 修改配置，重新编译运行
python test_return_probs.py cluster8_pivot4

# 再次修改配置，重新编译运行
python test_return_probs.py reduce_opt_v1

# 查看所有历史记录
python test_return_probs.py list

# 对比所有结果
python test_return_probs.py compare

# 只对比最近两次
python test_return_probs.py compare -1 -2

# 清空历史记录
python test_return_probs.py clear

# 精度测试
python test_return_probs.py acc 16
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
        print(f'{i=}, {k_i=}, {values[i, -1]=}, ')
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
    print(f"{batch_size=}, func1/func2: ratio={case1_ret[1]/case2_ret[1]:.2f}, avg_cost={case1_ret[1]:.3f}/{case2_ret[1]:.3f} ms,")

    # 返回结果用于批量测试
    return {
        'batch_size': batch_size,
        'func1_time': case1_ret[1],
        'func2_time': case2_ret[1],
        'ratio': case1_ret[1] / case2_ret[1]
    }

def test_top_k_top_p_sampling_acc(batch_size=1):
    # 测试配置
    vocab_size = 151936  # LLM词表大小
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

    gloden_ret = golden_impl(
        probs.contiguous(),
        top_ks,
        top_ps,
    )

    my_ret = top_k_top_p_filter_return_probs(
        probs.contiguous(),
        top_ks,
        top_ps,
        filter_apply_order="joint",
        check_nan=False,
    )

    mask = my_ret != gloden_ret
    diff_my = my_ret[mask]
    diff_golden = gloden_ret[mask]
    print(f'{torch.equal(my_ret, gloden_ret)=}, {diff_my=}, {diff_golden=},')

BENCHMARK_FILE = "benchmark_history.json"

def save_benchmark(config_name, results):
    """将 benchmark 结果追加到 JSON 文件"""
    import json
    import datetime
    import os

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        'name': config_name,
        'timestamp': timestamp,
        'results': results
    }

    # 加载现有数据或创建新列表
    history = []
    if os.path.exists(BENCHMARK_FILE):
        with open(BENCHMARK_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(entry)

    # 保存
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"[Saved] {config_name} @ {timestamp} -> {BENCHMARK_FILE}")
    return entry

def load_benchmarks():
    """从 JSON 文件加载所有历史 benchmark 结果"""
    import json
    import os

    if not os.path.exists(BENCHMARK_FILE):
        print(f"No benchmark history found at {BENCHMARK_FILE}")
        return []

    with open(BENCHMARK_FILE, 'r') as f:
        history = json.load(f)

    print(f"[Loaded] {len(history)} benchmark(s) from {BENCHMARK_FILE}")
    for i, entry in enumerate(history):
        print(f"  [{i}] {entry['name']} @ {entry['timestamp']}")
    return history

def clear_benchmarks():
    """清空历史记录"""
    import os
    if os.path.exists(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)
        print(f"[Cleared] {BENCHMARK_FILE}")

def run_batch_benchmark(config_name="default"):
    """运行批量测试，保存结果到文件，并渲染 ASCII 图表"""
    batch_sizes = list(range(1, 129, 4))
    results = []

    print(f"\n{'='*60}")
    print(f"Running benchmark: {config_name}")
    print(f"{'='*60}")

    for bs in batch_sizes:
        result = test_top_k_top_p_sampling_performance(bs)
        results.append(result)

    # 保存到 JSON 文件
    entry = save_benchmark(config_name, results)

    # 渲染 ASCII 图表
    render_speedup_chart(config_name, results)

    return entry

def render_speedup_chart(config_name, results):
    """渲染单次 benchmark 的 ASCII 图表"""
    print(f"\n{'='*60}")
    print(f"Speedup Ratio - {config_name}")
    print(f"{'='*60}")

    print(f"\n[func1/func2, lower is better, <1.0 means faster]")
    print(f"{'batch':<6} {'ratio':<8} {'bar (1.0 = same speed)'}")
    print("-" * 60)
    for r in results:
        ratio = r['ratio']
        if ratio <= 1.0:
            bar_len = int((1.0 - ratio) * 40)
            bar = "◀" + "━" * bar_len + "│"
            status = f"faster {(1.0/ratio - 1)*100:.0f}%"
        else:
            bar_len = int((ratio - 1.0) * 20)
            bar = "│" + "━" * bar_len + "▶"
            status = f"slower {(ratio - 1)*100:.0f}%"
        print(f"{r['batch_size']:<6} {ratio:<8.2f} {bar} ({status})")

    # 统计摘要
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Best ratio:  {min(r['ratio'] for r in results):.2f} at batch={min(results, key=lambda x: x['ratio'])['batch_size']}")
    print(f"  Worst ratio: {max(r['ratio'] for r in results):.2f} at batch={max(results, key=lambda x: x['ratio'])['batch_size']}")
    print(f"  Avg ratio:   {sum(r['ratio'] for r in results) / len(results):.2f}")
    faster_count = sum(1 for r in results if r['ratio'] < 1.0)
    print(f"  Faster in {faster_count}/{len(results)} cases")
    print(f"{'='*60}")


def compare_benchmarks(*indices):
    """
    对比历史 benchmark 结果

    用法:
      compare_benchmarks()       # 对比所有历史结果
      compare_benchmarks(-1, -2) # 对比最近两次
      compare_benchmarks(0, 2)   # 对比第 0 和第 2 次
    """
    history = load_benchmarks()
    if len(history) < 2:
        print("需要至少 2 个 benchmark 结果进行对比")
        return

    # 如果指定了索引，选择对应的 benchmark
    if indices:
        benchmark_results = [history[i] for i in indices]
    else:
        benchmark_results = history

    # 获取所有 batch sizes
    batch_sizes = [r['batch_size'] for r in benchmark_results[0]['results']]

    # 打印表头
    print(f"\n{'='*80}")
    print("Benchmark Comparison (ratio, lower is better)")
    print(f"{'='*80}")

    # 配置名称行
    header = f"{'batch':<8}"
    for br in benchmark_results:
        name = br['name'][:12]  # 截断名称
        header += f"{name:<14}"
    header += "best"
    print(header)
    print("-" * 80)

    # 数据行
    for i, bs in enumerate(batch_sizes):
        row = f"{bs:<8}"
        ratios = []
        for br in benchmark_results:
            ratio = br['results'][i]['ratio']
            ratios.append(ratio)
            # 标记最佳
            row += f"{ratio:<14.3f}"

        # 找出最佳配置
        best_idx = ratios.index(min(ratios))
        row += f"← {benchmark_results[best_idx]['name'][:10]}"
        print(row)

    # 汇总统计
    print("-" * 80)

    # 平均 ratio
    row = f"{'avg':<8}"
    avg_ratios = []
    for br in benchmark_results:
        avg = sum(r['ratio'] for r in br['results']) / len(br['results'])
        avg_ratios.append(avg)
        row += f"{avg:<14.3f}"
    best_idx = avg_ratios.index(min(avg_ratios))
    row += f"← {benchmark_results[best_idx]['name'][:10]}"
    print(row)

    # 最佳 ratio
    row = f"{'best':<8}"
    for br in benchmark_results:
        best = min(r['ratio'] for r in br['results'])
        row += f"{best:<14.3f}"
    print(row)

    # 最差 ratio
    row = f"{'worst':<8}"
    for br in benchmark_results:
        worst = max(r['ratio'] for r in br['results'])
        row += f"{worst:<14.3f}"
    print(row)

    print(f"{'='*80}")

    # ASCII 对比图
    print(f"\nComparison Chart (avg ratio per config)")
    print("-" * 60)
    max_avg = max(avg_ratios)
    for i, br in enumerate(benchmark_results):
        bar_len = int(avg_ratios[i] / max_avg * 30)
        bar = "█" * bar_len
        marker = " ★ BEST" if avg_ratios[i] == min(avg_ratios) else ""
        print(f"{br['name'][:20]:<20} {avg_ratios[i]:.3f} {bar}{marker}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    # 命令行参数解析
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "compare":
            # python test_return_probs.py compare        # 对比所有
            # python test_return_probs.py compare -1 -2  # 对比最近两次
            if len(sys.argv) > 2:
                indices = [int(x) for x in sys.argv[2:]]
                compare_benchmarks(*indices)
            else:
                compare_benchmarks()
        elif cmd == "list":
            # python test_return_probs.py list
            load_benchmarks()
        elif cmd == "clear":
            # python test_return_probs.py clear
            clear_benchmarks()
        elif cmd == "acc":
            # python test_return_probs.py acc [batch_size]
            bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            test_top_k_top_p_sampling_acc(bs)
        else:
            # python test_return_probs.py <config_name>
            run_batch_benchmark(cmd)
    else:
        print("Usage:")
        print("  python test_return_probs.py <config_name>  # Run benchmark with config name")
        print("  python test_return_probs.py compare        # Compare all benchmarks")
        print("  python test_return_probs.py compare -1 -2  # Compare specific indices")
        print("  python test_return_probs.py list           # List all benchmarks")
        print("  python test_return_probs.py clear          # Clear benchmark history")
        print("  python test_return_probs.py acc [bs]       # Run accuracy test")