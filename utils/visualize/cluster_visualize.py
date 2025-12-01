import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import re
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载Sentence-BERT模型
print("正在加载Sentence-BERT模型...")
start_time = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
load_time = time.time() - start_time
print(f"模型加载完成，耗时: {load_time:.2f}秒")

def clean_id(id_str):
    """去掉id中的数字，只保留类别名称"""
    return re.sub(r'\.\d+', '', id_str).lower()

def extract_triplets_from_json_file(file_path):
    """从JSON文件路径中提取GT和Pred的三元组"""
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None, None
    
    print(f"正在读取JSON文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 清理和解析JSON
        content = content.replace("'", '"')
        data = json.loads(content)
        
        print(f"成功加载JSON数据，image_id: {data.get('image_id', 'unknown')}")
        
    except Exception as e:
        print(f"读取或解析JSON文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 提取预测的三元组
    pred_triplets = []
    
    # 从response中提取关系
    response_str = data.get("response", "")
    
    # 解析关系部分 - 特别注意预测三元组的格式
    relationship_match = re.search(r'<RELATIONSHIP>({.*?})</RELATIONSHIP>', response_str, re.DOTALL)
    if relationship_match:
        try:
            relationship_data = json.loads(relationship_match.group(1))
            relationships = relationship_data.get("relationships", {})
            
            print(f"发现的关系类型: {list(relationships.keys())}")
            
            # 处理三个子关系类别
            relation_categories = {
                'spatial_relations': '空间关系',
                'possession_relations': '所属关系', 
                'semantic_relations': '语义关系'
            }
            
            for rel_category, rel_list in relationships.items():
                if rel_category in relation_categories:
                    category_name = relation_categories[rel_category]
                    print(f"\n处理 {rel_category} ({category_name})，数量: {len(rel_list)}")
                    
                    for i, rel in enumerate(rel_list):
                        # 清理id中的数字
                        subj_clean = clean_id(rel['subject'])
                        obj_clean = clean_id(rel['object'])
                        predicate = rel['predicate'].lower()
                        
                        triplet_text = f"{subj_clean} {predicate} {obj_clean}"
                        pred_triplets.append({
                            'text': triplet_text,
                            'subject': subj_clean,
                            'predicate': predicate,
                            'object': obj_clean,
                            'original_subject': rel['subject'],
                            'original_object': rel['object'],
                            'type': rel_category,
                            'category_name': category_name
                        })
                        
                        print(f"  {i+1}. {triplet_text} [{category_name}]")
                else:
                    print(f"跳过未知关系类型: {rel_category}")
                    
        except Exception as e:
            print(f"解析预测关系时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("未找到RELATIONSHIP标签")
    
    # 提取GT的三元组
    gt_triplets = []
    gt_relationships = data.get("gt_relationships", "[]")
    
    try:
        if isinstance(gt_relationships, str):
            gt_rels = json.loads(gt_relationships)
        else:
            gt_rels = gt_relationships
            
        print(f"\nGT关系数量: {len(gt_rels)}")
        
        for i, rel in enumerate(gt_rels):
            subj_clean = clean_id(rel['subject'])
            obj_clean = clean_id(rel['object'])
            predicate = rel['predicate'].lower()
            
            triplet_text = f"{subj_clean} {predicate} {obj_clean}"
            gt_triplets.append({
                'text': triplet_text,
                'subject': subj_clean,
                'predicate': predicate,
                'object': obj_clean,
                'original_subject': rel['subject'],
                'original_object': rel['object'],
                'type': 'gt_relation'
            })
            
            print(f"  GT {i+1}: {triplet_text}")
            
    except Exception as e:
        print(f"解析GT关系时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n提取结果统计:")
    print(f"预测三元组: {len(pred_triplets)} 个")
    print(f"GT三元组: {len(gt_triplets)} 个")
    
    return pred_triplets, gt_triplets

def encode_and_cluster_triplets(triplets, method='tsne', cluster_algo='dbscan', **cluster_kwargs):
    """
    对三元组进行编码和聚类
    
    参数:
    - triplets: 三元组列表
    - method: 降维方法 'tsne' 或 'pca'
    - cluster_algo: 聚类算法 'dbscan' 或 'kmeans'
    - cluster_kwargs: 其他聚类参数
    """
    if not triplets:
        print("没有三元组可处理")
        return None, None, None, None
    
    # 提取文本
    texts = [t['text'] for t in triplets]
    
    print(f"正在编码 {len(texts)} 个三元组...")
    start_time = time.time()
    embeddings = model.encode(texts)
    encode_time = time.time() - start_time
    print(f"编码完成，耗时: {encode_time:.2f}秒")
    print(f"嵌入向量形状: {embeddings.shape}")
    
    # 根据选择的算法进行聚类
    if len(triplets) > 1:
        if cluster_algo == 'dbscan':
            print("使用DBSCAN进行自动聚类...")
            
            # 从参数中获取或设置默认值
            eps = cluster_kwargs.get('eps', 0.5)
            min_samples = cluster_kwargs.get('min_samples', max(2, min(5, len(triplets) // 10)))
            auto_tune = cluster_kwargs.get('auto_tune', True)
            
            if auto_tune:
                # 自动调优DBSCAN参数
                best_clusters = None
                best_num_clusters = 0
                best_eps = eps
                
                # 尝试不同的eps值
                eps_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
                print(f"尝试eps参数: {eps_candidates}")
                
                for candidate_eps in eps_candidates:
                    dbscan = DBSCAN(eps=candidate_eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(embeddings)
                    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    
                    # 选择产生合理数量簇的参数
                    if 1 <= num_clusters <= min(15, len(triplets) // 2):
                        if num_clusters > best_num_clusters or (num_clusters == best_num_clusters and candidate_eps < best_eps):
                            best_num_clusters = num_clusters
                            best_clusters = clusters
                            best_eps = candidate_eps
                
                if best_clusters is None:
                    print("自动调优未找到合适参数，使用默认值")
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    best_clusters = dbscan.fit_predict(embeddings)
                    best_eps = eps
                else:
                    print(f"自动选择最佳eps: {best_eps}")
                
                clusters = best_clusters
            else:
                # 使用固定参数
                print(f"使用固定参数: eps={eps}, min_samples={min_samples}")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(embeddings)
            
        elif cluster_algo == 'kmeans':
            # 使用KMeans聚类
            n_clusters = cluster_kwargs.get('n_clusters', 3)
            n_clusters = min(n_clusters, len(triplets))
            print(f"使用KMeans聚类，簇数量: {n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
        else:
            raise ValueError(f"不支持的聚类算法: {cluster_algo}，请选择 'dbscan' 或 'kmeans'")
            
    else:
        clusters = [0] * len(triplets)
    
    # 统计聚类结果
    unique_clusters = np.unique(clusters)
    n_noise = np.sum(clusters == -1) if cluster_algo == 'dbscan' else 0
    n_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    print(f"{cluster_algo.upper()}聚类完成: 发现 {n_real_clusters} 个簇" + 
          (f", {n_noise} 个噪声点" if cluster_algo == 'dbscan' else ""))
    
    # 降维可视化
    if method == 'tsne':
        perplexity = min(30, max(5, len(embeddings) - 1))
        print(f"t-SNE perplexity: {perplexity}")
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"降维完成，形状: {reduced_embeddings.shape}")
    
    return embeddings, clusters, reduced_embeddings, texts

def visualize_clusters(triplets, reduced_embeddings, clusters, title, save_path=None, cluster_algo='dbscan'):
    """可视化聚类结果"""
    if reduced_embeddings is None or len(triplets) == 0:
        print(f"没有数据可可视化: {title}")
        return
    
    plt.figure(figsize=(16, 12))
    
    # 创建颜色映射
    unique_clusters = np.unique(clusters)
    
    # 为噪声点设置特殊颜色（仅DBSCAN）
    colors = []
    cluster_labels = []
    
    for i, cluster_id in enumerate(unique_clusters):
        if cluster_algo == 'dbscan' and cluster_id == -1:  # 噪声点
            colors.append('lightgray')
            cluster_labels.append('噪声点')
        else:
            colors.append(plt.cm.Set3(i / max(1, len(unique_clusters))))
            cluster_labels.append(f'簇 {cluster_id}')
    
    # 绘制散点图
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                   c=[colors[i]], label=cluster_labels[i], s=100, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
    
    # 添加标签（避免重叠）
    plotted_labels = set()
    label_positions = {}
    
    for i, (x, y) in enumerate(reduced_embeddings):
        # 噪声点不添加标签，避免图表过于拥挤
        if cluster_algo == 'dbscan' and clusters[i] == -1:
            continue
            
        label = triplets[i]['text']
        if len(label) > 25:
            label = label[:22] + "..."
        
        # 检查位置是否已有标签，避免重叠
        position_key = (round(x, 1), round(y, 1))
        if position_key not in label_positions:
            offset_x = 10
            offset_y = 10
        else:
            offset_x = 10 + label_positions[position_key] * 5
            offset_y = 10 + label_positions[position_key] * 5
            label_positions[position_key] += 1
        
        if label not in plotted_labels:
            plt.annotate(label, (x, y), xytext=(offset_x, offset_y), 
                        textcoords='offset points', fontsize=8, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3))
            plotted_labels.add(label)
            label_positions[position_key] = label_positions.get(position_key, 0) + 1
    
    n_noise = np.sum(clusters == -1) if cluster_algo == 'dbscan' else 0
    n_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    algo_name = "DBSCAN" if cluster_algo == 'dbscan' else "KMeans"
    noise_info = f", {n_noise}个噪声点" if cluster_algo == 'dbscan' else ""
    
    plt.title(f'{title} ({algo_name})\n(共{len(triplets)}个三元组, {n_real_clusters}个簇{noise_info})', 
              fontsize=16, pad=20)
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()

def analyze_cluster_statistics(triplets, clusters, data_type="数据", cluster_algo='dbscan'):
    """分析聚类统计信息"""
    print(f"\n{'='*60}")
    print(f"{data_type}聚类统计分析 ({cluster_algo.upper()})")
    print(f"{'='*60}")
    
    if not triplets:
        print("没有三元组可分析")
        return
    
    unique_clusters = np.unique(clusters)
    n_noise = np.sum(clusters == -1) if cluster_algo == 'dbscan' else 0
    n_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    print(f"总三元组数: {len(triplets)}")
    print(f"聚类数量: {n_real_clusters} 个")
    if cluster_algo == 'dbscan':
        print(f"噪声点数量: {n_noise} 个")
    
    # 总体统计
    if 'type' in triplets[0]:
        type_counts = {}
        for triplet in triplets:
            t_type = triplet.get('type', 'unknown')
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
        print(f"关系类型分布: {type_counts}")
    
    # 处理噪声点（仅DBSCAN）
    if cluster_algo == 'dbscan' and -1 in unique_clusters:
        noise_indices = np.where(clusters == -1)[0]
        noise_triplets = [triplets[i] for i in noise_indices]
        print(f"\n噪声点 (包含 {len(noise_triplets)} 个三元组):")
        
        # 统计噪声点的谓词频率
        predicates = [t['predicate'] for t in noise_triplets]
        predicate_counts = {}
        for pred in predicates:
            predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
        print(f"  谓词分布: {dict(sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        # 显示代表性噪声三元组
        print("  代表性噪声三元组:")
        for i, triplet in enumerate(noise_triplets[:5]):
            type_info = f" [{triplet.get('category_name', '')}]" if 'category_name' in triplet else ""
            print(f"    {i+1}. {triplet['text']}{type_info}")
    
    # 处理正常簇
    for cluster_id in unique_clusters:
        if cluster_algo == 'dbscan' and cluster_id == -1:  # 跳过噪声点，已经处理过了
            continue
            
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_triplets = [triplets[i] for i in cluster_indices]
        
        print(f"\n簇 {cluster_id} (包含 {len(cluster_triplets)} 个三元组):")
        
        # 统计谓词频率
        predicates = [t['predicate'] for t in cluster_triplets]
        predicate_counts = {}
        for pred in predicates:
            predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
        print(f"  谓词分布: {dict(sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        # 如果是预测数据，显示关系类型
        if 'category_name' in cluster_triplets[0]:
            categories = [t['category_name'] for t in cluster_triplets]
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            print(f"  关系类型: {category_counts}")
        
        # 显示代表性三元组
        print("  代表性三元组:")
        for i, triplet in enumerate(cluster_triplets[:3]):
            type_info = f" [{triplet.get('category_name', '')}]" if 'category_name' in triplet else ""
            print(f"    {i+1}. {triplet['text']}{type_info}")

def create_comparison_visualization(pred_triplets, gt_triplets, pred_clusters, gt_clusters, 
                                  pred_reduced, gt_reduced, pred_algo='dbscan', gt_algo='dbscan'):
    """创建预测和GT的对比可视化"""
    if pred_reduced is None or gt_reduced is None:
        print("无法创建对比可视化：缺少数据")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 预测数据可视化
    unique_pred_clusters = np.unique(pred_clusters)
    pred_colors = []
    pred_labels = []
    
    for i, cluster_id in enumerate(unique_pred_clusters):
        if pred_algo == 'dbscan' and cluster_id == -1:
            pred_colors.append('lightgray')
            pred_labels.append('噪声点')
        else:
            pred_colors.append(plt.cm.Set3(i / max(1, len(unique_pred_clusters))))
            pred_labels.append(f'簇 {cluster_id}')
    
    for i, cluster_id in enumerate(unique_pred_clusters):
        mask = pred_clusters == cluster_id
        ax1.scatter(pred_reduced[mask, 0], pred_reduced[mask, 1],
                   c=[pred_colors[i]], label=pred_labels[i], s=80, alpha=0.7)
    
    n_pred_noise = np.sum(pred_clusters == -1) if pred_algo == 'dbscan' else 0
    n_pred_clusters = len(unique_pred_clusters) - (1 if -1 in unique_pred_clusters else 0)
    
    pred_algo_name = "DBSCAN" if pred_algo == 'dbscan' else "KMeans"
    ax1.set_title(f'预测三元组聚类 ({pred_algo_name})\n({len(pred_triplets)}个三元组, {n_pred_clusters}个簇' + 
                 (f', {n_pred_noise}个噪声点)' if pred_algo == 'dbscan' else ')'), fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # GT数据可视化
    unique_gt_clusters = np.unique(gt_clusters)
    gt_colors = []
    gt_labels = []
    
    for i, cluster_id in enumerate(unique_gt_clusters):
        if gt_algo == 'dbscan' and cluster_id == -1:
            gt_colors.append('lightgray')
            gt_labels.append('噪声点')
        else:
            gt_colors.append(plt.cm.Set3(i / max(1, len(unique_gt_clusters))))
            gt_labels.append(f'簇 {cluster_id}')
    
    for i, cluster_id in enumerate(unique_gt_clusters):
        mask = gt_clusters == cluster_id
        ax2.scatter(gt_reduced[mask, 0], gt_reduced[mask, 1],
                   c=[gt_colors[i]], label=gt_labels[i], s=80, alpha=0.7)
    
    n_gt_noise = np.sum(gt_clusters == -1) if gt_algo == 'dbscan' else 0
    n_gt_clusters = len(unique_gt_clusters) - (1 if -1 in unique_gt_clusters else 0)
    
    gt_algo_name = "DBSCAN" if gt_algo == 'dbscan' else "KMeans"
    ax2.set_title(f'GT三元组聚类 ({gt_algo_name})\n({len(gt_triplets)}个三元组, {n_gt_clusters}个簇' + 
                 (f', {n_gt_noise}个噪声点)' if gt_algo == 'dbscan' else ')'), fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 主执行函数
def main():
    # 从文件路径读取JSON数据
    json_file_path = "/root/R1-SGG/gspo-cot-recall-soft-weight-relation-psg/107907.json"  # 请替换为实际的JSON文件路径
    
    # 聚类算法选择（可以在这里修改）
    PRED_CLUSTER_ALGO = 'dbscan'  # 可选: 'dbscan' 或 'kmeans'
    GT_CLUSTER_ALGO = 'dbscan'    # 可选: 'dbscan' 或 'kmeans'
    
    # 算法特定参数
    CLUSTER_KWARGS = {
        'dbscan': {
            'eps': 0.775,
            'min_samples': 1,
            'auto_tune': False  # 是否自动调优参数
        },
        'kmeans': {
            'n_clusters': 3  # KMeans的簇数量
        }
    }
    
    print(f"聚类算法配置:")
    print(f"预测数据: {PRED_CLUSTER_ALGO.upper()}")
    print(f"GT数据: {GT_CLUSTER_ALGO.upper()}")
    
    # 提取三元组
    pred_triplets, gt_triplets = extract_triplets_from_json_file(json_file_path)
    
    if pred_triplets is None:
        print("无法读取数据，程序退出")
        return
    
    print(f"\n{'='*50}")
    print("开始处理预测三元组")
    print(f"{'='*50}")
    
    # 处理预测三元组
    if pred_triplets:
        pred_embeddings, pred_clusters, pred_reduced, pred_texts = encode_and_cluster_triplets(
            pred_triplets, 
            method='tsne', 
            cluster_algo=PRED_CLUSTER_ALGO,
            **CLUSTER_KWARGS[PRED_CLUSTER_ALGO]
        )
        
        # 可视化预测聚类
        visualize_clusters(pred_triplets, pred_reduced, pred_clusters, 
                          "预测三元组聚类可视化", "pred_clusters.png", cluster_algo=PRED_CLUSTER_ALGO)
        
        # 分析预测聚类统计
        analyze_cluster_statistics(pred_triplets, pred_clusters, "预测", cluster_algo=PRED_CLUSTER_ALGO)
    
    print(f"\n{'='*50}")
    print("开始处理GT三元组")
    print(f"{'='*50}")
    
    # 处理GT三元组
    if gt_triplets:
        gt_embeddings, gt_clusters, gt_reduced, gt_texts = encode_and_cluster_triplets(
            gt_triplets,
            method='tsne', 
            cluster_algo=GT_CLUSTER_ALGO,
            **CLUSTER_KWARGS[GT_CLUSTER_ALGO]
        )
        
        # 可视化GT聚类
        visualize_clusters(gt_triplets, gt_reduced, gt_clusters, 
                          "GT三元组聚类可视化", "gt_clusters.png", cluster_algo=GT_CLUSTER_ALGO)
        
        # 分析GT聚类统计
        analyze_cluster_statistics(gt_triplets, gt_clusters, "GT", cluster_algo=GT_CLUSTER_ALGO)
        
        # 创建对比可视化
        if pred_triplets and gt_triplets:
            create_comparison_visualization(pred_triplets, gt_triplets, 
                                          pred_clusters, gt_clusters,
                                          pred_reduced, gt_reduced,
                                          pred_algo=PRED_CLUSTER_ALGO, gt_algo=GT_CLUSTER_ALGO)
    
    print(f"\n{'='*50}")
    print("处理完成")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()