import openai
from typing import List
import re

class CoTGenerator:
    """基于GPT-4的思维链生成器"""
    def __init__(self, api_key):
        openai.api_key = api_key
        self.physical_rules = [
            r'力\s*[<>≤≥]\s*\d+\.?\d*\s*N',
            r'超过\d+度',
            r'(遮挡|碰撞|倾倒)'
        ]
    
    def generate(self, instruction: str) -> List[str]:
        """生成伪CoT数据"""
        prompt = f"""
        请将以下机器人操作指令分解为分步推理过程：
        指令：{instruction}
        按照以下要求生成：
        1. 包含3-5个步骤
        2. 每个步骤包含置信度估计（0-1之间）
        3. 使用中文技术术语
        4. 包含物理约束检查
        
        示例：
        1. 识别场景中所有红色物体 [置信度 0.95]
        2. 检测目标物体的支撑平面 [置信度 0.87]
        3. 规划无碰撞运动轨迹 [置信度 0.78]
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_steps = response.choices[0].message.content.split('\n')
        return self._filter_steps(raw_steps)
    
    def _filter_steps(self, steps: List[str]) -> List[str]:
        """物理规则过滤"""
        valid_steps = []
        for step in steps:
            if not any(re.search(pattern, step) for pattern in self.physical_rules):
                valid_steps.append(step)
        return valid_steps
