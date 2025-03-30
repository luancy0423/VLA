import json
import hashlib
import numpy as np
from functools import lru_cache
from diskcache import Cache

class SemanticCache:
    def __init__(self):
        self.memory_cache = LRUCache(maxsize=1000)
        self.disk_cache = Cache('cache_dir')
        self.disk_cache.expire()  # 启动过期清理
        
    def query(self, feature_vector, metadata):
        # 生成复合键
        mem_key = self._generate_memory_key(feature_vector)
        disk_key = self._generate_disk_key(metadata)
        
        # 多级缓存查询
        if result := self.memory_cache.get(mem_key):
            return result
        if result := self.disk_cache.get(disk_key):
            self.memory_cache[mem_key] = result  # 填充内存缓存
            return result
        return None
    
    def update(self, feature_vector, metadata, action_data):
        # 生成键
        mem_key = self._generate_memory_key(feature_vector)
        disk_key = self._generate_disk_key(metadata)
        
        # 更新缓存
        self.memory_cache[mem_key] = action_data
        self.disk_cache.set(disk_key, action_data)
        
    def _generate_memory_key(self, vector):
        # 基于特征向量的快速哈希
        return hashlib.sha256(vector.tobytes()).hexdigest()
    
    def _generate_disk_key(self, metadata):
        # 基于语义属性的结构化键
        key_data = {
            'category': metadata.get('category'),
            'material': metadata.get('material'),
            'size_group': self._size_group(metadata['size'])
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _size_group(self, size):
        # 离散化尺寸
        volume = np.prod(size)
        if volume < 0.001: return 'small'
        elif volume < 0.1: return 'medium'
        else: return 'large'

class LRUCache:
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self.cache = {}
        self.order = []
        
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def __setitem__(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.maxsize:
            del_key = self.order.pop(0)
            del self.cache[del_key]
        self.cache[key] = value
        self.order.append(key)

# 示例用法
if __name__ == "__main__":
    cache = SemanticCache()
    
    # 模拟特征向量和元数据
    dummy_feature = np.random.randn(256)
    metadata = {
        'category': 'cup',
        'material': 'ceramic',
        'size': [0.1, 0.1, 0.15]
    }
    
    # 缓存更新
    cache.update(dummy_feature, metadata, {'action': [0.2, 0.1, ...]})
    
    # 缓存查询
    result = cache.query(dummy_feature, metadata)
    print("缓存命中:", result is not None)
