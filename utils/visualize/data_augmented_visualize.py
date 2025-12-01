import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os
from datetime import datetime

def load_relationship_data(file_path):
    """加载关系JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def extract_predicates(data):
    """从数据中提取所有谓词及其出现频率"""
    all_predicates = []
    
    for sample in data:
        if 'relationships' in sample:
            relationships = sample['relationships']
            
            # 检查三种关系类别
            for category in ['spatial_relations', 'possession_relations', 'semantic_relations']:
                if category in relationships and isinstance(relationships[category], list):
                    for rel in relationships[category]:
                        if isinstance(rel, dict) and 'predicate' in rel:
                            all_predicates.append(rel['predicate'])
    
    return Counter(all_predicates)

def save_frequency_txt(predicate_counts, save_dir='./visualizations'):
    """保存频率从大到小排序的txt文件"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算总关系数
    total_relations = sum(predicate_counts.values())
    
    # 按频率从大到小排序
    sorted_predicates = predicate_counts.most_common()
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file = os.path.join(save_dir, f'predicate_frequency_rank_{timestamp}.txt')
    
    # 写入txt文件
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("谓词频率排名（从高到低）\n")
        f.write("=" * 60 + "\n")
        f.write(f"总关系数: {total_relations}\n")
        f.write(f"唯一谓词数: {len(predicate_counts)}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'排名':<6} {'谓词':<20} {'频次':<8} {'百分比':<10}\n")
        f.write("-" * 50 + "\n")
        
        for rank, (predicate, count) in enumerate(sorted_predicates, 1):
            percentage = (count / total_relations) * 100
            f.write(f"{rank:<6} {predicate:<20} {count:<8} {percentage:.2f}%\n")
    
    print(f"频率排名txt文件已保存: {txt_file}")
    return txt_file

def plot_predicate_percentage_bar_chart(predicate_counts, top_n=50, save_dir='./visualizations'):
    """绘制谓词百分比频率条形图"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取频率最高的top_n个谓词
    top_predicates = predicate_counts.most_common(top_n)
    predicates, counts = zip(*top_predicates)
    
    # 计算总关系数
    total_relations = sum(predicate_counts.values())
    
    # 计算百分比频率
    percentages = [count / total_relations * 100 for count in counts]
    
    # 设置图形大小
    plt.figure(figsize=(20, 10))
    
    # 创建条形图
    bars = plt.bar(range(len(predicates)), percentages, color='skyblue', alpha=0.8)
    
    # 设置坐标轴标签和标题
    plt.xlabel('谓词', fontsize=12)
    plt.ylabel('百分比频率 (%)', fontsize=12)
    plt.title(f'Top {top_n} 谓词百分比频率分布', fontsize=14, fontweight='bold')
    
    # 设置x轴刻度
    plt.xticks(range(len(predicates)), predicates, rotation=45, ha='right', fontsize=10)
    
    # 设置y轴范围，留出空间显示标签
    plt.ylim(0, max(percentages) * 1.15)
    
    # 在条形上添加百分比标签
    for i, (percentage, count) in enumerate(zip(percentages, counts)):
        plt.text(i, percentage + max(percentages)*0.01, 
                f'{percentage:.1f}%', 
                ha='center', va='bottom', fontsize=9, rotation=0)
    
    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f'predicate_percentage_frequency_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"谓词百分比频率条形图已保存: {filename}")
    
    plt.close()
    
    return predicates, percentages, filename

def generate_statistics(predicate_counts, data):
    """生成统计信息"""
    total_relations = sum(predicate_counts.values())
    unique_predicates = len(predicate_counts)
    total_samples = len(data)
    
    print("=" * 50)
    print("关系数据统计摘要")
    print("=" * 50)
    print(f"总样本数: {total_samples}")
    print(f"总关系数: {total_relations}")
    print(f"唯一谓词数: {unique_predicates}")
    print(f"平均每个样本的关系数: {total_relations/total_samples:.2f}")
    
    return {
        'total_samples': total_samples,
        'total_relations': total_relations,
        'unique_predicates': unique_predicates,
        'avg_relations_per_sample': total_relations / total_samples
    }

def main():
    """主函数"""
    # 文件路径
    file_path = "/root/R1-SGG/relationship_augmentation_psg.json"  # 修改为您的文件路径
    save_dir = "./predicate_visualizations_psg"  # 可视化结果保存目录
    
    # 加载数据
    print("正在加载数据...")
    data = load_relationship_data(file_path)
    
    if not data:
        print("无法加载数据，请检查文件路径")
        return
    
    print(f"成功加载 {len(data)} 个样本")
    
    # 提取谓词频率
    print("正在分析谓词频率...")
    predicate_counts = extract_predicates(data)
    
    # 生成统计信息
    stats = generate_statistics(predicate_counts, data)
    
    # 保存频率排名txt文件
    print("正在生成频率排名txt文件...")
    txt_file = save_frequency_txt(predicate_counts, save_dir=save_dir)
    
    # 绘制谓词百分比频率条形图
    print("正在生成谓词百分比频率条形图...")
    predicates, percentages, image_file = plot_predicate_percentage_bar_chart(
        predicate_counts, 
        top_n=min(50, len(predicate_counts)),  # 最多显示50个谓词
        save_dir=save_dir
    )
    
    # 显示前10个最常出现的谓词
    print(f"\n前10个最常出现的谓词:")
    for i, (predicate, percentage) in enumerate(zip(predicates[:10], percentages[:10])):
        count = predicate_counts[predicate]
        print(f"{i+1}. {predicate}: {count}次 ({percentage:.2f}%)")
    
    # 保存统计结果到文件
    output_stats = {
        'overall_statistics': stats,
        'top_50_predicates': [
            {'predicate': predicate, 'count': predicate_counts[predicate], 'percentage': percentage}
            for predicate, percentage in zip(predicates, percentages)
        ],
        'saved_files': {
            'image': image_file,
            'txt': txt_file
        }
    }
    
    # 保存统计结果
    stats_file = os.path.join(save_dir, 'predicate_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(output_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计结果已保存: {stats_file}")
    print(f"可视化图表已保存: {image_file}")
    print(f"频率排名文件已保存: {txt_file}")
    
    # 显示长尾分布信息
    long_tail = len([p for p in predicate_counts.values() if p == 1])
    total_unique = len(predicate_counts)
    rare_predicates = len([p for p in predicate_counts.values() if p < 5])
    
    print(f"\n长尾分布分析:")
    print(f"只出现1次的谓词数量: {long_tail} ({long_tail/total_unique*100:.1f}% 的唯一谓词)")
    print(f"出现次数少于5次的谓词数量: {rare_predicates} ({rare_predicates/total_unique*100:.1f}% 的唯一谓词)")
    
    # 显示txt文件内容预览
    print(f"\n频率排名文件预览 (前20个):")
    print("=" * 50)
    sorted_predicates = predicate_counts.most_common(20)
    print(f"{'排名':<6} {'谓词':<20} {'频次':<8} {'百分比':<10}")
    print("-" * 50)
    for rank, (predicate, count) in enumerate(sorted_predicates, 1):
        percentage = (count / sum(predicate_counts.values())) * 100
        print(f"{rank:<6} {predicate:<20} {count:<8} {percentage:.2f}%")

if __name__ == "__main__":
    main()