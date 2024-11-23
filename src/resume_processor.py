# data_processor.py
import json

class ResumeProcessor:
    def __init__(self, resume_path):
        self.resume_path = resume_path
    
    def parse_resume(self):
        # 解析简历，提取关键信息
        with open(self.resume_path, 'r') as f:
            resume_data = json.load(f)
        
        # 结构化简历内容
        structured_resume = {
            "basic_info": resume_data.get("basic_info", {}),
            "work_experience": resume_data.get("work_experience", []),
            "skills": resume_data.get("skills", []),
            "education": resume_data.get("education", [])
        }
        
        return structured_resume