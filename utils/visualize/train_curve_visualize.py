import re
import matplotlib.pyplot as plt
import json
import numpy as np

def parse_log_file(file_path):
    """解析日志文件，提取指标数据"""
    steps = []
    category_f1_rewards = []  # 修正：改为category_f1_rewards
    box_acc_rewards = []
    edge_rewards = []
    node_recall_rewards = []
    edge_diversity_rewards = []
    
    step_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # 使用正则表达式匹配所有字典格式的数据
        pattern = r"\{[^}]+\}"
        matches = re.findall(pattern, content)
        
        for match in matches:
            try:
                # 清理数据并转换为字典
                data_str = match.replace("'", "\"")
                data = json.loads(data_str)
                
                # 检查是否包含loss（表示一个训练步）
                if 'loss' in data:
                    step_count += 1
                    
                    # 提取所需指标 - 支持多种可能的指标名称
                    # Category F1 Reward (原Node Accuracy Reward)
                    category_f1 = data.get('rewards/stage2_category_reward/mean', 
                                      data.get('rewards/stage1_category_reward/mean', 
                                      data.get('rewards/category_f1_reward/mean', 0)))  # 新增可能的键名
                    
                    # Box Accuracy Reward  
                    box_acc = data.get('rewards/stage2_node_box_reward/mean', 
                                     data.get('rewards/stage3_node_box_reward/mean',
                                     data.get('rewards/box_acc_reward/mean', 0)))  # 新增可能的键名
                    
                    # Node Recall Reward
                    node_recall = data.get('rewards/stage2_node_recall_reward/mean',
                                         data.get('rewards/stage3_node_recall_reward/mean',
                                         data.get('rewards/node_recall_reward/mean', 0)))  # 新增可能的键名
                    
                    # Edge Reward
                    edge_reward = data.get('rewards/edge_reward/mean', 
                                         data.get('rewards/stage3_edge_fine_reward/mean',
                                         data.get('rewards/edge_acc_reward/mean', 0)))  # 新增可能的键名
    
                    # Edge Diversity Reward
                    edge_diversity = data.get('rewards/stage3_edge_coarse_reward/mean', 
                                           data.get('rewards/edge_coarse_reward/mean', 0))  # 新增可能的键名
                    
                    steps.append(step_count)
                    category_f1_rewards.append(category_f1)  # 修正：使用正确的变量名
                    box_acc_rewards.append(box_acc)
                    edge_rewards.append(edge_reward)
                    node_recall_rewards.append(node_recall)
                    edge_diversity_rewards.append(edge_diversity)
                    
            except json.JSONDecodeError:
                continue
    
    return steps, category_f1_rewards, box_acc_rewards, edge_rewards, node_recall_rewards, edge_diversity_rewards

def smooth_data(data, window_size=100):
    """对数据进行滑动平均平滑"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i+1]
        smoothed.append(sum(window_data) / len(window_data))
    
    return smoothed

def downsample_data(steps, data, interval=100):
    """每interval步采样一次数据"""
    if len(steps) <= interval:
        return steps, data
    
    downsampled_steps = []
    downsampled_data = []
    
    for i in range(0, len(steps), interval):
        downsampled_steps.append(steps[i])
        downsampled_data.append(data[i])
    
    return downsampled_steps, downsampled_data

def plot_and_save_metrics(file_paths, labels=None, output_file="training_metrics.png", smooth_window=100):
    """绘制并保存图表到文件"""
    if labels is None:
        labels = [f"File {i+1}" for i in range(len(file_paths))]
    
    # 设置非交互式后端
    plt.switch_backend('Agg')
    
    # 创建图表 - 调整为5个子图
    plt.figure(figsize=(15, 25))  # 增加高度以适应5个子图
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 增加颜色数量
    line_styles = ['-', '--', '-.', ':', '-']
    line_widths = [2, 2.5, 2, 2.2, 2.3]
    
    all_data = {}
    
    for i, file_path in enumerate(file_paths):
        steps, category_f1, box_acc, edge_reward, node_recall, edge_diversity = parse_log_file(file_path)  # 修正变量名
        
        if not steps:
            print(f"在文件 {file_path} 中没有找到有效数据")
            continue
        
        # 存储原始数据
        all_data[labels[i]] = {
            'steps': steps,
            'category_f1': category_f1,  # 修正键名
            'box_acc': box_acc,
            'edge_reward': edge_reward,
            'node_recall': node_recall,
            'edge_diversity': edge_diversity
        }
        
        print(f"\n📁 {labels[i]} 数据统计:")
        print(f"   总步数: {len(steps)}")
        print(f"   Category F1 Reward范围: {min(category_f1):.3f} - {max(category_f1):.3f}")  # 修正
        print(f"   Box Acc Reward范围: {min(box_acc):.3f} - {max(box_acc):.3f}")
        print(f"   Edge Reward范围: {min(edge_reward):.3f} - {max(edge_reward):.3f}")
        print(f"   Node Recall Reward范围: {min(node_recall):.3f} - {max(node_recall):.3f}")
        print(f"   Edge Diversity Reward范围: {min(edge_diversity):.3f} - {max(edge_diversity):.3f}")
        
        # 应用滑动平均
        smoothed_category_f1 = smooth_data(category_f1, smooth_window)  # 修正变量名
        smoothed_box_acc = smooth_data(box_acc, smooth_window)
        smoothed_edge_reward = smooth_data(edge_reward, smooth_window)
        smoothed_node_recall = smooth_data(node_recall, smooth_window)
        smoothed_edge_diversity = smooth_data(edge_diversity, smooth_window)
        
        # 降采样用于显示
        if len(steps) > 200:
            display_steps, display_category_f1 = downsample_data(steps, smoothed_category_f1, 100)  # 修正
            _, display_box_acc = downsample_data(steps, smoothed_box_acc, 100)
            _, display_edge_reward = downsample_data(steps, smoothed_edge_reward, 100)
            _, display_node_recall = downsample_data(steps, smoothed_node_recall, 100)
            _, display_edge_diversity = downsample_data(steps, smoothed_edge_diversity, 100)
        else:
            display_steps, display_category_f1 = steps, smoothed_category_f1  # 修正
            display_box_acc, display_edge_reward = smoothed_box_acc, smoothed_edge_reward
            display_node_recall = smoothed_node_recall
            display_edge_diversity = smoothed_edge_diversity
        
        # 绘制五个子图
        # 1. Category F1 Reward (原Node Accuracy Reward)
        plt.subplot(5, 1, 1)
        plt.plot(display_steps, display_category_f1,  # 修正
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_widths[i % len(line_widths)],
                label=labels[i],
                alpha=0.8)
        
        # 2. Box Accuracy Reward
        plt.subplot(5, 1, 2)
        plt.plot(display_steps, display_box_acc, 
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_widths[i % len(line_widths)],
                label=labels[i],
                alpha=0.8)
        
        # 3. Edge Reward
        plt.subplot(5, 1, 3)
        plt.plot(display_steps, display_edge_reward, 
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_widths[i % len(line_widths)],
                label=labels[i],
                alpha=0.8)
        
        # 4. Node Recall Reward
        plt.subplot(5, 1, 4)
        plt.plot(display_steps, display_node_recall, 
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_widths[i % len(line_widths)],
                label=labels[i],
                alpha=0.8)
        
        # 5. Edge Diversity Reward
        plt.subplot(5, 1, 5)
        plt.plot(display_steps, display_edge_diversity, 
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_widths[i % len(line_widths)],
                label=labels[i],
                alpha=0.8)
    
    # 设置第一个子图 - Category F1 Reward
    plt.subplot(5, 1, 1)
    plt.ylabel('Category F1 Reward', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Category F1 Reward Comparison (100-step moving average)', fontsize=11)
    
    # 设置第二个子图 - Box Accuracy Reward
    plt.subplot(5, 1, 2)
    plt.ylabel('Box Accuracy Reward', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Box Accuracy Reward Comparison (100-step moving average)', fontsize=11)
    
    # 设置第三个子图 - Edge Reward
    plt.subplot(5, 1, 3)
    plt.ylabel('Edge Reward', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Edge Reward Comparison (100-step moving average)', fontsize=11)
    
    # 设置第四个子图 - Node Recall Reward
    plt.subplot(5, 1, 4)
    plt.ylabel('Node Recall Reward', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Node Recall Reward Comparison (100-step moving average)', fontsize=11)
    
    # 设置第五个子图 - Edge Diversity Reward
    plt.subplot(5, 1, 5)
    plt.ylabel('Edge Diversity Reward', fontsize=12, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.title('Edge Diversity Reward Comparison (100-step moving average)', fontsize=11)
    
    plt.suptitle('Training Metrics Comparison with 100-Step Moving Average', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # 保存图表
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 图表已保存为: {output_file}")
    
    # 打印统计信息
    print_statistics(all_data)

def print_statistics(all_data):
    """打印详细的统计信息"""
    print("\n" + "="*80)
    print("详细统计信息 (基于100步滑动平均后数据)")
    print("="*80)
    
    for label, data in all_data.items():
        steps = data['steps']
        category_f1 = smooth_data(data['category_f1'], 100)  # 修正
        box_acc = smooth_data(data['box_acc'], 100)
        edge_reward = smooth_data(data['edge_reward'], 100)
        node_recall = smooth_data(data['node_recall'], 100)
        edge_diversity = smooth_data(data['edge_diversity'], 100)
        
        if steps:
            print(f"\n📊 {label} 统计信息:")
            print(f"   📈 总步数: {len(steps)}")
            print(f"   🔵 Category F1 Reward - 最终平滑值: {category_f1[-1]:.3f}")  # 修正
            print(f"   🟠 Box Acc Reward - 最终平滑值: {box_acc[-1]:.3f}")
            print(f"   🟢 Edge Reward - 最终平滑值: {edge_reward[-1]:.3f}")
            print(f"   🔴 Node Recall Reward - 最终平滑值: {node_recall[-1]:.3f}")
            print(f"   🟣 Edge Diversity Reward - 最终平滑值: {edge_diversity[-1]:.3f}")
            
            # 显示变化趋势
            if len(steps) > 100:
                # Category F1趋势
                start_avg = sum(category_f1[:100]) / min(100, len(category_f1))  # 修正
                end_avg = sum(category_f1[-100:]) / min(100, len(category_f1))  # 修正
                trend = "↑ 上升" if end_avg > start_avg else "↓ 下降" if end_avg < start_avg else "→ 平稳"
                print(f"   📊 Category F1 趋势: {trend} ({start_avg:.3f} → {end_avg:.3f})")  # 修正
                
                # Box Acc趋势
                start_box = sum(box_acc[:100]) / min(100, len(box_acc))
                end_box = sum(box_acc[-100:]) / min(100, len(box_acc))
                trend_box = "↑ 上升" if end_box > start_box else "↓ 下降" if end_box < start_box else "→ 平稳"
                print(f"   📊 Box Acc 趋势: {trend_box} ({start_box:.3f} → {end_box:.3f})")
                
                # Node Recall趋势
                start_recall = sum(node_recall[:100]) / min(100, len(node_recall))
                end_recall = sum(node_recall[-100:]) / min(100, len(node_recall))
                trend_recall = "↑ 上升" if end_recall > start_recall else "↓ 下降" if end_recall < start_recall else "→ 平稳"
                print(f"   📊 Node Recall 趋势: {trend_recall} ({start_recall:.3f} → {end_recall:.3f})")
                
                # Edge Diversity趋势
                start_diversity = sum(edge_diversity[:100]) / min(100, len(edge_diversity))
                end_diversity = sum(edge_diversity[-100:]) / min(100, len(edge_diversity))
                trend_diversity = "↑ 上升" if end_diversity > start_diversity else "↓ 下降" if end_diversity < start_diversity else "→ 平稳"
                print(f"   📊 Edge Diversity 趋势: {trend_diversity} ({start_diversity:.3f} → {end_diversity:.3f})")

def debug_file_content(file_path):
    """调试函数：查看文件内容结构"""
    print(f"\n🔍 调试文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找前几个包含loss的字典
            pattern = r"\{[^}]+\}"
            matches = re.findall(pattern, content)
            
            found_count = 0
            for match in matches[:5]:  # 增加检查数量到5个
                if "'loss'" in match or "\"loss\"" in match:
                    found_count += 1
                    print(f"找到第{found_count}个数据点:")
                    # 简化显示，只显示关键信息
                    data_str = match.replace("'", "\"")
                    try:
                        data = json.loads(data_str)
                        # 显示所有包含reward的键
                        reward_keys = [k for k in data.keys() if any(term in k.lower() for term in ['reward', 'recall', 'diversity', 'category', 'box', 'edge'])]
                        print(f"  相关指标 ({len(reward_keys)}个):")
                        for key in reward_keys:
                            value = data.get(key, 'N/A')
                            if isinstance(value, (int, float)):
                                print(f"    {key}: {value:.4f}")
                            else:
                                print(f"    {key}: {value}")
                    except Exception as e:
                        print(f"   解析失败: {e}")
            
            print(f"总共找到 {len(matches)} 个可能的字典结构")
            
    except Exception as e:
        print(f"读取文件失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 替换为您的实际文件路径
    file1_path = "/root/R1-SGG/gspo_2b_train_weight_soft_relation_psg.log"
    file2_path = "/root/R1-SGG/gspo_2b_train_weight_soft_relation_data_psg.log"
    
    file_paths = [file1_path, file2_path]
    labels = ["Model_1", "Model_2"]
    
    # 先调试文件内容
    for file_path in file_paths:
        debug_file_content(file_path)
    
    try:
        # 绘制并保存图表
        plot_and_save_metrics(file_paths, labels, "training_metrics_comparison_gspo.png", smooth_window=100)
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        import traceback
        traceback.print_exc()