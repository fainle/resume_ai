import os
import json
import numpy as np

from langchain.embeddings import OpenAIEmbeddings
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema
from dotenv import load_dotenv

load_dotenv()


class ResumeVectorizer:
    def __init__(self):
        connections.connect(host='localhost', port='19530')
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self._init_collection()

    def _init_collection(self):
        fields =[
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='resume_vector', dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="basic_info", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="professional_summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="skills", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="work_experience", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="education", dtype=DataType.VARCHAR, max_length=65535)
        ]
        # 定义结构
        schema = CollectionSchema(fields=fields, description="简历基础结构")

        #创建集合
        self.collection = Collection(name='resumes', schema=schema)

        #创建索引
        index_params = {
            "metric_type": "L2",
            'index_type': "IVF_FLAT",
            "params": {'nlist': 1024}
        }

        self.collection.create_index(field_name='resume_vector', index_params=index_params)

    def store_resume(self, resume_text, parsed_data):
        try:
            # 生成嵌入向量
            vector = self.embeddings.embed_query(resume_text)
            
            # 确保向量是 numpy array 并且是正确的类型
            # vector = np.array(vector, dtype=np.float32)
            
            # 关键修改: 不要将向量包装在列表中
            data = {
                "resume_vector": vector,  # 直接使用 numpy array，不要包装在列表中
                "content": str(resume_text),
                "basic_info": json.dumps(parsed_data.get("basic_info", {})),
                "professional_summary": json.dumps(parsed_data.get("professional_summary", [])),
                "skills": json.dumps(parsed_data.get("skills", {})),
                "work_experience": json.dumps(parsed_data.get("work_experience", [])),
                "education": json.dumps(parsed_data.get("education", {})),
            }
        
            # 调试信息
            # print(f"Vector shape: {vector.shape}")
            # print(f"Vector type: {type(vector)}")
            # print(f"Sample vector values: {vector[:5]}")
            
            # 插入数据
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            return insert_result

        except Exception as e:
            print(f"Error storing resume: {str(e)}")
            print(f"Data type of vector: {type(vector)}")
            print(f"Vector shape: {getattr(vector, 'shape', 'No shape available')}")
            raise

    def search_similar_resumes(self, query_text, top_k=5):
        """搜索相似简历"""
        # 生成查询向量
        query_vector = self.embeddings.embed_query(query_text)
        
        # 执行搜索
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="resume_vector",
            param=search_params,
            limit=top_k,
            output_fields=["content"]
        )
        
        return results[0]  # 返回第一个查询的结果
    
    def clean_up(self):
        """清理连接"""
        self.collection.release()
        connections.disconnect("default")