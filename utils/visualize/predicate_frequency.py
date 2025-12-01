import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import os
from datetime import datetime
import pandas as pd
from datasets import load_dataset, load_from_disk

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PredicateFrequencyVisualizer:
    """谓词频率可视化分析器"""
    
    def __init__(self, dataset, output_dir="predicate_analysis"):
        self.dataset = dataset
        self.output_dir = output_dir
        self.predicate_counter = Counter()
        self.subject_counter = Counter()
        self.object_counter = Counter()
        self.triplet_counter = Counter()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_dataset(self):
        """分析数据集中的谓词频率"""
        print("开始分析数据集谓词频率...")
        
        total_relations = 0
        for example in tqdm(self.dataset, desc="分析样本"):
            try:
                # 解析关系数据
                if 'relationships' in example:
                    rels = example['relationships']
                    if not isinstance(rels, (list, tuple)):
                        rels = json.loads(rels)
                    
                    for rel in rels:
                        predicate = rel.get('predicate', '').strip()
                        subject = rel.get('subject', '').strip()
                        obj = rel.get('object', '').strip()
                        
                        if predicate and predicate != '__background__':
                            # 统计谓词频率
                            self.predicate_counter[predicate] += 1
                            
                            # 统计主语和宾语类别
                            if '.' in subject:
                                subj_category = subject.split('.')[0]
                                self.subject_counter[subj_category] += 1
                            if '.' in obj:
                                obj_category = obj.split('.')[0]
                                self.object_counter[obj_category] += 1
                            
                            # 统计三元组频率
                            triplet = f"{subject} - {predicate} - {obj}"
                            self.triplet_counter[triplet] += 1
                            
                            total_relations += 1
                            
            except Exception as e:
                continue
        
        print(f"分析完成！共处理 {total_relations} 个关系")
        return total_relations
    
    def get_statistics(self):
        """获取详细的统计信息"""
        total_relations = sum(self.predicate_counter.values())
        unique_predicates = len(self.predicate_counter)
        unique_subjects = len(self.subject_counter)
        unique_objects = len(self.object_counter)
        unique_triplets = len(self.triplet_counter)
        
        # 计算长尾分布
        frequencies = list(self.predicate_counter.values())
        sorted_freq = sorted(frequencies, reverse=True)
        cumulative = np.cumsum(sorted_freq)
        total = cumulative[-1] if len(cumulative) > 0 else 0
        
        # 找到达到不同累积百分比的谓词数量
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        threshold_results = {}
        
        for threshold in thresholds:
            if total > 0:
                idx = np.argmax(cumulative >= total * threshold)
                threshold_results[f'top_{int(threshold*100)}%'] = idx + 1
            else:
                threshold_results[f'top_{int(threshold*100)}%'] = 0
        
        statistics = {
            'total_relations': total_relations,
            'unique_predicates': unique_predicates,
            'unique_subjects': unique_subjects,
            'unique_objects': unique_objects,
            'unique_triplets': unique_triplets,
            'most_common_predicates': self.predicate_counter.most_common(20),
            'least_common_predicates': self.predicate_counter.most_common()[-30:][::-1],
            'all_predicates': self.predicate_counter.most_common(),  # 新增：所有谓词及其频率
            'predicate_diversity': unique_predicates / max(1, total_relations),
            'long_tail_distribution': threshold_results,
            'coverage_analysis': {
                'top_10_predicates_coverage': sum([freq for _, freq in self.predicate_counter.most_common(10)]) / max(1, total_relations),
                'top_20_predicates_coverage': sum([freq for _, freq in self.predicate_counter.most_common(20)]) / max(1, total_relations),
                'top_50_predicates_coverage': sum([freq for _, freq in self.predicate_counter.most_common(50)]) / max(1, total_relations),
            }
        }
        
        return statistics
    
    def plot_predicate_frequency_distribution(self, top_k=50):
        """绘制谓词频率分布图"""
        if not self.predicate_counter:
            print("没有谓词数据可分析")
            return
        
        # 准备数据
        predicates = [p[0] for p in self.predicate_counter.most_common(top_k)]
        frequencies = [p[1] for p in self.predicate_counter.most_common(top_k)]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. 频率柱状图
        bars = ax1.bar(range(len(predicates)), frequencies, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Predicates')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Top {top_k} Most Frequent Predicates')
        ax1.set_xticks(range(len(predicates)))
        ax1.set_xticklabels(predicates, rotation=45, ha='right', fontsize=8)
        
        # 在柱子上添加数值
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(frequencies)*0.01,
                    f'{freq}', ha='center', va='bottom', fontsize=7, rotation=0)
        
        # 2. 对数尺度频率分布
        ax2.bar(range(len(predicates)), frequencies, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_xlabel('Predicates')
        ax2.set_ylabel('Frequency (Log Scale)')
        ax2.set_title('Predicate Frequency Distribution (Log Scale)')
        ax2.set_yscale('log')
        ax2.set_xticks(range(len(predicates)))
        ax2.set_xticklabels(predicates, rotation=45, ha='right', fontsize=8)
        
        # 3. 累积分布图
        all_frequencies = sorted(self.predicate_counter.values(), reverse=True)
        cumulative_percentage = np.cumsum(all_frequencies) / sum(all_frequencies) * 100
        
        ax3.plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage, 
                'b-', linewidth=2, label='Cumulative Percentage')
        ax3.set_xlabel('Number of Predicates')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Distribution of Predicates')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 标记关键点
        key_points = [50, 80, 90, 95, 99]
        for point in key_points:
            idx = np.argmax(cumulative_percentage >= point)
            if idx < len(cumulative_percentage):
                ax3.plot(idx + 1, cumulative_percentage[idx], 'ro', markersize=6)
                ax3.annotate(f'{point}%', (idx + 1, cumulative_percentage[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 4. 长尾分析图
        ranks = range(1, len(all_frequencies) + 1)
        ax4.loglog(ranks, all_frequencies, 'g-', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Predicate Rank (Log Scale)')
        ax4.set_ylabel('Frequency (Log Scale)')
        ax4.set_title('Long-tail Distribution (Zipf\'s Law)')
        ax4.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(all_frequencies) > 1:
            z = np.polyfit(np.log(ranks), np.log(all_frequencies), 1)
            p = np.poly1d(z)
            ax4.loglog(ranks, np.exp(p(np.log(ranks))), 'r--', alpha=0.8, 
                      label=f'Trend: y ~ x^{z[0]:.2f}')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/predicate_frequency_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"谓词频率分布图已保存至: {self.output_dir}/predicate_frequency_distribution.png")
    
    def plot_predicate_coverage_analysis(self):
        """绘制谓词覆盖分析图"""
        if not self.predicate_counter:
            return
        
        # 计算累积覆盖
        frequencies = sorted(self.predicate_counter.values(), reverse=True)
        cumulative = np.cumsum(frequencies)
        total = cumulative[-1]
        cumulative_percentage = cumulative / total * 100
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 覆盖百分比 vs 谓词数量
        predicate_counts = range(1, len(cumulative_percentage) + 1)
        ax1.plot(predicate_counts, cumulative_percentage, 'b-', linewidth=2)
        ax1.set_xlabel('Number of Predicates')
        ax1.set_ylabel('Cumulative Coverage (%)')
        ax1.set_title('Predicate Coverage Analysis')
        ax1.grid(True, alpha=0.3)
        
        # 标记关键覆盖点
        coverage_points = [10, 20, 50, 100, 200]
        for point in coverage_points:
            if point <= len(cumulative_percentage):
                coverage = cumulative_percentage[point-1]
                ax1.plot(point, coverage, 'ro', markersize=6)
                ax1.annotate(f'{point} preds\n{coverage:.1f}%', 
                           (point, coverage), xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 2. 帕累托图（前N个谓词的覆盖比例）
        top_n = min(100, len(frequencies))
        top_predicates_coverage = []
        
        for n in range(1, top_n + 1):
            coverage = sum(frequencies[:n]) / total * 100
            top_predicates_coverage.append(coverage)
        
        ax2.bar(range(1, top_n + 1), top_predicates_coverage, 
               color='orange', alpha=0.6, edgecolor='darkorange')
        ax2.set_xlabel('Top N Predicates')
        ax2.set_ylabel('Coverage Percentage (%)')
        ax2.set_title(f'Coverage by Top {top_n} Predicates')
        ax2.grid(True, alpha=0.3)
        
        # 标记关键点
        for n in [10, 20, 50]:
            if n <= top_n:
                coverage = top_predicates_coverage[n-1]
                ax2.plot(n, coverage, 'ro', markersize=6)
                ax2.annotate(f'{coverage:.1f}%', (n, coverage), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/predicate_coverage_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"谓词覆盖分析图已保存至: {self.output_dir}/predicate_coverage_analysis.png")
    
    def plot_category_analysis(self):
        """绘制主语和宾语类别分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 主语类别分析
        if self.subject_counter:
            top_subjects = self.subject_counter.most_common(20)
            subjects, counts = zip(*top_subjects)
            
            ax1.barh(range(len(subjects)), counts, color='lightgreen', alpha=0.7)
            ax1.set_yticks(range(len(subjects)))
            ax1.set_yticklabels(subjects)
            ax1.set_xlabel('Frequency')
            ax1.set_title('Top 20 Most Frequent Subject Categories')
            ax1.invert_yaxis()  # 最高的在顶部
        
        # 宾语类别分析
        if self.object_counter:
            top_objects = self.object_counter.most_common(20)
            objects, counts = zip(*top_objects)
            
            ax2.barh(range(len(objects)), counts, color='lightcoral', alpha=0.7)
            ax2.set_yticks(range(len(objects)))
            ax2.set_yticklabels(objects)
            ax2.set_xlabel('Frequency')
            ax2.set_title('Top 20 Most Frequent Object Categories')
            ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/category_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别分析图已保存至: {self.output_dir}/category_analysis.png")
    
    def plot_heatmap_analysis(self, top_n=30):
        """绘制谓词热力图分析"""
        if not self.predicate_counter:
            return
        
        # 获取top N谓词
        top_predicates = [p[0] for p in self.predicate_counter.most_common(top_n)]
        
        # 创建频率矩阵（这里简化处理，实际可以根据需要构建更复杂的关系矩阵）
        freq_matrix = np.zeros((top_n, top_n))
        
        # 这里可以扩展为实际的共现矩阵分析
        # 目前先创建一个简单的对角线矩阵用于演示
        for i, pred in enumerate(top_predicates):
            freq_matrix[i, i] = self.predicate_counter[pred]
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(freq_matrix, 
                   xticklabels=top_predicates,
                   yticklabels=top_predicates,
                   cmap='YlOrRd', 
                   annot=False,  # 数据太多时不显示数值
                   fmt='.0f',
                   cbar_kws={'label': 'Frequency'})
        
        plt.title(f'Predicate Frequency Heatmap (Top {top_n})')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/predicate_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"谓词热力图已保存至: {self.output_dir}/predicate_heatmap.png")
    
    def generate_comprehensive_report(self):
        """生成全面的分析报告"""
        stats = self.get_statistics()
        
        report_path = f'{self.output_dir}/predicate_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PREDICATE FREQUENCY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Relations: {stats['total_relations']:,}\n")
            f.write(f"Unique Predicates: {stats['unique_predicates']:,}\n")
            f.write(f"Unique Subject Categories: {stats['unique_subjects']:,}\n")
            f.write(f"Unique Object Categories: {stats['unique_objects']:,}\n")
            f.write(f"Unique Triplets: {stats['unique_triplets']:,}\n")
            f.write(f"Predicate Diversity: {stats['predicate_diversity']:.4f}\n\n")
            
            f.write("COVERAGE ANALYSIS:\n")
            for key, value in stats['coverage_analysis'].items():
                f.write(f"  {key}: {value:.3%}\n")
            f.write("\n")
            
            f.write("LONG TAIL DISTRIBUTION:\n")
            for key, value in stats['long_tail_distribution'].items():
                f.write(f"  {key}: {value} predicates\n")
            f.write("\n")
            
            f.write("TOP 20 MOST FREQUENT PREDICATES:\n")
            for i, (predicate, count) in enumerate(stats['most_common_predicates'], 1):
                percentage = (count / stats['total_relations']) * 100
                f.write(f"  {i:2d}. {predicate:<20} {count:>6,} ({percentage:6.2f}%)\n")
            f.write("\n")
            
            f.write("BOTTOM 20 LEAST FREQUENT PREDICATES:\n")
            for i, (predicate, count) in enumerate(stats['least_common_predicates'], 1):
                percentage = (count / stats['total_relations']) * 100
                f.write(f"  {i:2d}. {predicate:<20} {count:>6,} ({percentage:8.4f}%)\n")
            f.write("\n")
            
            # 新增：输出所有谓词及其频率
            f.write("="*80 + "\n")
            f.write("ALL PREDICATES AND THEIR FREQUENCIES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total unique predicates: {len(stats['all_predicates']):,}\n\n")
            
            for i, (predicate, count) in enumerate(stats['all_predicates'], 1):
                percentage = (count / stats['total_relations']) * 100
                f.write(f"{i:4d}. {predicate:<25} {count:>8,} ({percentage:7.3f}%)\n")
        
        print(f"分析报告已保存至: {report_path}")
        return report_path
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始完整的谓词频率分析...")
        
        # 1. 分析数据
        total_relations = self.analyze_dataset()
        
        if total_relations == 0:
            print("没有找到关系数据，分析终止")
            return
        
        # 2. 生成统计报告
        stats = self.get_statistics()
        
        # 3. 生成所有可视化图表
        self.plot_predicate_frequency_distribution(top_k=50)
        self.plot_predicate_coverage_analysis()
        self.plot_category_analysis()
        self.plot_heatmap_analysis(top_n=30)
        
        # 4. 生成详细报告
        report_path = self.generate_comprehensive_report()
        
        print("\n" + "="*60)
        print("分析完成！")
        print(f"输出文件保存在: {self.output_dir}/")
        print(f"总关系数: {total_relations:,}")
        print(f"唯一谓词数: {stats['unique_predicates']:,}")
        print(f"前10个谓词覆盖: {stats['coverage_analysis']['top_10_predicates_coverage']:.1%}")
        print(f"前20个谓词覆盖: {stats['coverage_analysis']['top_20_predicates_coverage']:.1%}")
        print("="*60)

# 使用示例
def analyze_predicate_frequency(dataset_path_or_name, output_dir="predicate_analysis"):
    """
    分析数据集的谓词频率
    
    Args:
        dataset_path_or_name: 数据集路径或名称
        output_dir: 输出目录
    """
    # 加载数据集
    print(f"加载数据集: {dataset_path_or_name}")
    try:
        if os.path.exists(dataset_path_or_name):
            dataset = load_from_disk(dataset_path_or_name)
        else:
            dataset = load_dataset(dataset_path_or_name)
        
        # 确保是训练集
        if isinstance(dataset, dict):
            dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
        else:
            dataset = dataset['train'] if hasattr(dataset, 'keys') and 'train' in dataset.keys() else dataset
    
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 创建分析器并运行分析
    analyzer = PredicateFrequencyVisualizer(dataset, output_dir)
    analyzer.run_complete_analysis()

# 直接运行分析的函数
if __name__ == "__main__":
    # 使用方法1: 直接指定数据集
    # analyze_predicate_frequency("your/dataset/path")
    
    # 使用方法2: 在代码中调用
    # 这里需要您提供实际的数据集路径
    dataset_path = "JosephZ/psg_train_sg"  # 请修改为实际路径
    
    try:
        analyze_predicate_frequency(dataset_path, "train_predicate_analysis_results-psg")
    except Exception as e:
        print(f"分析过程中出错: {e}")
        print("请检查数据集路径是否正确")