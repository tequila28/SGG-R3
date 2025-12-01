import json
import re
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

from rl.prompt_gallery_cot import get_psg_categories, get_vg150_categories, VG150_PREDICATES, PSG_REL_CATEGORIES, PSG_OBJ_CATEGORIES, VG150_OBJ_CATEGORIES

class PromptBuilder:
    def __init__(self, dataset_type="vg"):
        self.dataset_type = dataset_type
    
    def get_relation_categories(self):
        """根据数据集类型返回对应的关系分类"""
        if self.dataset_type == "psg":
            categories = get_psg_categories()
            return (categories["spatial_relations"], 
                    categories["static_action_relations"], 
                    categories["dynamic_action_relations"])
        else:  # vg dataset
            categories = get_vg150_categories()
            return (categories["spatial_relations"], 
                    categories["possession_relations"], 
                    categories["interaction_relations"])
    
    def get_predicates(self):
        """根据数据集类型返回对应的谓词列表"""
        if self.dataset_type == "psg":
            return PSG_REL_CATEGORIES
        else:  # vg dataset (default)
            return VG150_PREDICATES
    
    def get_object_categories(self):
        """根据数据集类型返回对应的物体类别列表"""
        if self.dataset_type == "psg":
            return PSG_OBJ_CATEGORIES
        else:  # vg dataset (default)
            return VG150_OBJ_CATEGORIES[1:]  # 排除背景类别

    def extract_all_relations_from_response(self, response_text, available_objects):
        """
        从模型响应中提取所有合法三元组，使用正则表达式直接匹配
        返回: 所有合法三元组的列表
        """
        all_relations = []
        valid_predicates = set(self.get_predicates())
        available_object_ids = {obj['id'] for obj in available_objects}
        
        print(f"开始提取关系，响应长度: {len(response_text)}")
        
        # 使用正则表达式直接提取所有三元组
        relations = self._extract_relations_with_regex(response_text, available_object_ids, valid_predicates)
        all_relations.extend(relations)
        print(f"从正则表达式提取到 {len(relations)} 个关系")
        
        # 去重
        unique_relations = self._remove_duplicate_relations(all_relations)
        print(f"去重后得到 {len(unique_relations)} 个唯一关系")
        
        return unique_relations

    def _extract_relations_with_regex(self, text, available_object_ids, valid_predicates):
        """使用正则表达式直接从文本中提取三元组"""
        relations = []
        
        # 定义多种可能的三元组模式
        patterns = [
            # JSON格式: {"subject": "xxx", "predicate": "xxx", "object": "xxx"}
            r'{\s*"subject"\s*:\s*"([^"]+)"\s*,\s*"predicate"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"\s*}',
            r"{\s*'subject'\s*:\s*'([^']+)'\s*,\s*'predicate'\s*:\s*'([^']+)'\s*,\s*'object'\s*:\s*'([^']+)'\s*}",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    # 提取三个组成部分
                    groups = match.groups()
                    if len(groups) >= 3:
                        subject = groups[0].strip().strip('"').strip("'")
                        predicate = groups[1].strip().strip('"').strip("'")
                        obj = groups[2].strip().strip('"').strip("'")
                        
                        # 创建关系字典
                        relation = {
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj
                        }
                        
                        # 验证关系有效性
                        if self._is_valid_relation(relation, available_object_ids, valid_predicates):
                            relations.append(relation)
                            print(f"找到有效关系: {subject} - {predicate} - {obj}")
                        else:
                            print(f"无效关系被过滤: {subject} - {predicate} - {obj}")
                            
                except Exception as e:
                    print(f"解析匹配时出错: {e}, 匹配内容: {match.group()}")
                    continue
        
        return relations

    def _is_valid_relation(self, relation, available_object_ids, valid_predicates):
        """验证关系是否有效"""
        if not isinstance(relation, dict):
            return False
        
        required_keys = ["subject", "predicate", "object"]
        if not all(key in relation for key in required_keys):
            return False
        
        subject = relation["subject"]
        predicate = relation["predicate"]
        obj = relation["object"]
        
        # 检查是否为空
        if not subject or not predicate or not obj:
            return False
        
        # 检查物体ID和谓词的有效性
        if (subject not in available_object_ids or 
            obj not in available_object_ids or
            predicate not in valid_predicates):
            return False
        
        return True

    def _remove_duplicate_relations(self, relations):
        """去除重复的关系"""
        unique_relations = []
        seen = set()
        
        for rel in relations:
            key = (rel["subject"], rel["predicate"], rel["object"])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations

    def merge_and_categorize_relations(self, model_relations, existing_relations, available_objects):
        """
        合并模型生成的关系和现有关系，并进行分类
        返回: 分类后的结果字典
        """
        # 合并所有关系
        all_relations = model_relations + existing_relations
        
        # 去重
        unique_relations = self._remove_duplicate_relations(all_relations)
        
        print(f"合并后关系统计:")
        print(f"  - 模型生成关系: {len(model_relations)}")
        print(f"  - 现有关系: {len(existing_relations)}")
        print(f"  - 合并去重后: {len(unique_relations)}")
        
        # 分类
        categorized_result = self._categorize_relations(unique_relations)
        
        return categorized_result

    def _categorize_relations(self, relations):
        """将关系分类到正确的类别中"""
        # 获取谓词分类
        spatial_predicates, cat2_predicates, cat3_predicates = self._get_predicate_sets()
        
        # 初始化结果结构
        if self.dataset_type == "psg":
            result = {
                "spatial_relations": [],
                "static_action_relations": [],
                "dynamic_action_relations": []
            }
        else:
            result = {
                "spatial_relations": [],
                "possession_relations": [],
                "interaction_relations": []
            }
        
        # 分类关系
        for rel in relations:
                
            predicate = rel["predicate"]
            
            if predicate in spatial_predicates:
                result["spatial_relations"].append(rel)
            elif predicate in cat2_predicates:
                if self.dataset_type == "psg":
                    result["static_action_relations"].append(rel)
                else:
                    result["possession_relations"].append(rel)
            elif predicate in cat3_predicates:
                if self.dataset_type == "psg":
                    result["dynamic_action_relations"].append(rel)
                else:
                    result["interaction_relations"].append(rel)
            else:
                print(f"警告: 谓词 '{predicate}' 不属于任何预定义类别")
        
        # 统计
        for category, rels in result.items():
            print(f"  - {category}: {len(rels)} 个关系")
        
        return result

    def _get_predicate_sets(self):
        """获取谓词集合"""
        spatial, cat2, cat3 = self.get_relation_categories()
        return set(spatial), set(cat2), set(cat3)

    # 保留原有的预处理方法
    def preprocess_objects_and_relationships(self, objects, relationships):
        """预处理物体和关系：将物体类别转换为类内计数格式"""
        obj_map = {}
        
        categories = []
        seen = set()
        for obj in objects:
            category_name = obj["id"].split(".")[0]
            if category_name not in seen:
                seen.add(category_name)
                categories.append({"id": category_name})
        
        category_to_objects = {}
        for obj in objects:
            category = obj["id"].split(".")[0]
            if category not in category_to_objects:
                category_to_objects[category] = []
            category_to_objects[category].append(obj)
        
        ordered_objects = []
        category_counters = {cat["id"]: 1 for cat in categories}
        
        for category in categories:
            cat_name = category["id"]
            if cat_name in category_to_objects:
                for obj in category_to_objects[cat_name]:
                    original_id = obj["id"]
                    new_id = f"{cat_name}.{category_counters[cat_name]}"
                    category_counters[cat_name] += 1
                    obj_map[original_id] = new_id
                    
                    new_obj = obj.copy()
                    new_obj["id"] = new_id
                    ordered_objects.append(new_obj)
        
        new_relationships = []
        for rel in relationships:
            subject_old = rel["subject"]
            object_old = rel["object"]
            predicate = rel["predicate"]
            
            if subject_old in obj_map and object_old in obj_map:
                new_relationships.append({
                    "subject": obj_map[subject_old],
                    "predicate": predicate,
                    "object": obj_map[object_old]
                })
            else:
                print(f"Warning: Object ID {subject_old} or {object_old} not found in object mapping")
        
        return ordered_objects, new_relationships

    def preprocess_sample(self, sample):
        """预处理单个样本"""
        try:
            objects = json.loads(sample['objects'])
            relationships = json.loads(sample['relationships'])
            
            new_objects, new_relationships = self.preprocess_objects_and_relationships(objects, relationships)
            
            new_sample = sample.copy()
            new_sample['objects'] = json.dumps(new_objects, ensure_ascii=False)
            new_sample['relationships'] = json.dumps(new_relationships, ensure_ascii=False)
            
            return new_sample
            
        except Exception as e:
            print(f"Error preprocessing sample: {e}")
            return sample

    def construct_relationship_augmentation_prompt(self, sample):
        """构造关系增强prompt"""
        image = sample["image"].convert('RGB')
        preprocessed_sample = self.preprocess_sample(sample)
        objects = json.loads(preprocessed_sample['objects'])
        relationships = json.loads(preprocessed_sample['relationships'])
        
        spatial_relations, category2_relations, category3_relations = self.get_relation_categories()
        
        if self.dataset_type == "psg":
            category2_name = "static_action_relations"
            category3_name = "dynamic_action_relations"
            category2_display = "Static Action Relations"
            category3_display = "Dynamic Action Relations"
        else:
            category2_name = "possession_relations"
            category3_name = "interaction_relations"
            category2_display = "Possession Relations"
            category3_display = "Interaction Relations"
        
        prompt = f'''
    # Two-Stage Visual Relation Augmentation

    Execute a structured, two-stage relation augmentation analysis for the provided image with pre-annotated objects. Your goal is to generate plausible relations. 

    ## Available Relationship Categories
    - **Spatial Relations:** {json.dumps(spatial_relations)}
    - **{category2_display}:** {json.dumps(category2_relations)}
    - **{category3_display}:** {json.dumps(category3_relations)}

    ## Pre-annotated Objects ({len(objects)} objects, USE EXACT IDs):
    Each object includes:
    - id: Unique identifier (e.g., "person.1", "car.2")
    - box: Bounding box coordinates [x1, y1, x2, y2] representing object position
    {json.dumps(objects, indent=2)}

    ## Stage 1: Scene Context Understanding
    - **Task:** Create a comprehensive natural language description for the image that incorporates ALL pre-annotated objects with their exact IDs.
    - **Requirements:**
        - Naturally include all {len(objects)} objects using their specific IDs.
        - Describe spatial arrangements, potential interactions, and scene context.
        - Be descriptive but concise.

    ## Stage 2: Multi-Category Relation Extraction
    - **Task:** Based on the global understanding from Stage 1, independently analyze relations in three categories, examining all object pairs within each category.
    - ***Three Relation Categories:**
        1. **Spatial Relations:** Analyze object pairs for new spatial/topological relations.
        2. **{category2_display}:** Analyze for new {'static action/pose' if self.dataset_type == 'psg' else 'ownership/composition/part-whole'} relations.
        3. **{category3_display}:** Analyze for new {'dynamic action/interaction' if self.dataset_type == 'psg' else 'action-oriented/functional'} relations.
    - **Requirements:**
        - Sequentially analyze each object in Stage 2 as the subject against all other objects in the scene to extract meaningful relations.
        - All relations must be between objects in pre-annotated objects.
        - All relation predicates MUST belong to their respective predefined category.

    ## Output Format Requirements
    Return ONLY a JSON object with this exact structure:

    {{
    "caption": "A detailed scene description naturally incorporating all object IDs like person.1, car.1, etc.",
    "spatial_relations": [
        {{"subject": "object.id", "predicate": "spatial_relations", "object": "object.id"}}
      
    ],
    "{category2_name}": [
        {{"subject": "object.id", "predicate": {category2_name}, "object": "object.id"}}
        
    ],
    "{category3_name}": [
        {{"subject": "object.id", "predicate": {category3_name}, "object": "object.id"}}
    ]
    }}

    Generate the two-stage relationship augmentation analysis, focusing on creating meaningful relationships:'''

        return prompt, image

    def parse_relationship_response(self, response_text, existing_relationships, available_objects):
        """
        新的解析流程：使用正则表达式提取所有三元组，合并现有关系，分类输出
        """
        print("=" * 60)
        print("开始解析模型响应...")
        
        try:
            # 1. 从模型响应中提取所有合法三元组
            model_relations = self.extract_all_relations_from_response(response_text, available_objects)
            
            # 2. 解析现有关系
            existing_relations = []
            if existing_relationships:
                if isinstance(existing_relationships, str):
                    existing_relations = json.loads(existing_relationships)
                else:
                    existing_relations = existing_relationships
            
            # 3. 合并模型关系和现有关系，并进行分类
            final_result = self.merge_and_categorize_relations(model_relations, existing_relations, available_objects)
            
            # 4. 添加caption（如果存在）
            caption = self._extract_caption(response_text)
            if caption:
                final_result["caption"] = caption
                print(f"提取到caption: {caption[:100]}...")
            
            # 5. 统计最终结果
            total_relations = sum(len(rels) for rels in final_result.values() if isinstance(rels, list))
            print(f"解析完成! 总共生成 {total_relations} 个关系")
            
            return final_result
            
        except Exception as e:
            print(f"解析失败: {e}")
            print(f"响应预览: {response_text[:500]}...")
            
            # 回退方案：仅使用现有关系
            print("使用回退方案：仅分类现有关系")
            return self.fallback_categorize_relationships(
                existing_relationships if isinstance(existing_relationships, list) else json.loads(existing_relationships), 
                available_objects
            )

    def _extract_caption(self, text):
        """提取caption字段"""
        patterns = [
            r'"caption"\s*:\s*"([^"]+)"',
            r"'caption'\s*:\s*'([^']+)'"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def fallback_categorize_relationships(self, existing_relationships, available_objects):
        """回退方案：仅分类现有关系"""
        if not existing_relationships:
            return self._create_empty_result()
        
        # 确保existing_relationships是列表
        if isinstance(existing_relationships, str):
            try:
                existing_relations = json.loads(existing_relationships)
            except:
                existing_relations = []
        else:
            existing_relations = existing_relationships
        
        print("回退方案：分类现有关系")
        return self._categorize_relations(existing_relations, available_objects)

    def _create_empty_result(self):
        """创建空的结果结构"""
        if self.dataset_type == "psg":
            return {
                "spatial_relations": [],
                "static_action_relations": [],
                "dynamic_action_relations": []
            }
        else:
            return {
                "spatial_relations": [],
                "possession_relations": [],
                "interaction_relations": []
            }