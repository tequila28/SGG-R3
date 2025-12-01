import matplotlib.pyplot as plt
import math

def calculate_weight(frequency_percent):
    freq = max(frequency_percent, 0.001)
    inverse_freq = 1.0 / freq
    log_inverse = math.log(inverse_freq)
    min_log = math.log(1.0/29.49)
    max_log = math.log(1.0/0.0015)
    normalized = (log_inverse - min_log) / (max_log - min_log)
    weight = 2.0 + normalized * 8.0
    return round(weight, 2)

PREDICATE_WEIGHTS = {
    'on': calculate_weight(29.49), 'has': calculate_weight(17.98),
    'wearing': calculate_weight(11.96), 'of': calculate_weight(9.15),
    'in': calculate_weight(5.66), 'near': calculate_weight(4.56),
    'behind': calculate_weight(3.17), 'with': calculate_weight(3.16),
    'holding': calculate_weight(2.48), 'above': calculate_weight(1.63),
    'wears': calculate_weight(1.25), 'sitting on': calculate_weight(1.19),
    'under': calculate_weight(1.10), 'in front of': calculate_weight(0.98),
    'riding': calculate_weight(0.85), 'standing on': calculate_weight(0.58),
    'at': calculate_weight(0.45), 'attached to': calculate_weight(0.39),
    'carrying': calculate_weight(0.37), 'walking on': calculate_weight(0.34),
    'over': calculate_weight(0.2470), 'for': calculate_weight(0.2452),
    'belonging to': calculate_weight(0.2028), 'hanging from': calculate_weight(0.1939),
    'looking at': calculate_weight(0.1805), 'parked on': calculate_weight(0.1735),
    'laying on': calculate_weight(0.1687), 'and': calculate_weight(0.1560),
    'covering': calculate_weight(0.1408), 'eating': calculate_weight(0.1308),
    'between': calculate_weight(0.1296), 'part of': calculate_weight(0.1259),
    'along': calculate_weight(0.1256), 'covered in': calculate_weight(0.1215),
    'using': calculate_weight(0.1129), 'watching': calculate_weight(0.1103),
    'on back of': calculate_weight(0.0973), 'to': calculate_weight(0.0906),
    'lying on': calculate_weight(0.0776), 'walking in': calculate_weight(0.0717),
    'mounted on': calculate_weight(0.0661), 'against': calculate_weight(0.0591),
    'across': calculate_weight(0.0520), 'from': calculate_weight(0.0505),
    'growing on': calculate_weight(0.0501), 'painted on': calculate_weight(0.0457),
    'made of': calculate_weight(0.0301), 'playing': calculate_weight(0.0193),
    'says': calculate_weight(0.0093), 'flying in': calculate_weight(0.0015),
}

# 按权重排序
sorted_weights = sorted(PREDICATE_WEIGHTS.items(), key=lambda x: x[1], reverse=True)
predicates = [item[0] for item in sorted_weights]
weights = [item[1] for item in sorted_weights]

# 创建图形
plt.figure(figsize=(16, 8))
bars = plt.bar(predicates, weights, color='skyblue')

# 设置图表属性
plt.xlabel('谓词', fontsize=12)
plt.ylabel('权重', fontsize=12)
plt.title('谓词权重分布', fontsize=14)
plt.xticks(rotation=90, fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整边距
plt.tight_layout()

# 保存图片
output_path = 'predicate_weights_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"图表已保存至: {output_path}")
