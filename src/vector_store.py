from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    connections, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType
)
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class ResumeVectorStore:
    def __init__(self, host="localhost", port="19530"):
        # Read OpenAI Key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Connect to Milvus
        connections.connect(host=host, port=port)
        
        # Define vector collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields)
        
        # Create or get collection
        self.collection_name = "resume_collection"
        
        try:
            # Try to get the existing collection
            self.collection = Collection(self.collection_name)
            
            # Check if index exists, if not, create it
            if not self.collection.has_index():
                # Create index
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index("embedding", index_params)
        
        except Exception:
            # Collection doesn't exist, so create it
            self.collection = Collection(self.collection_name, schema)
            
            # Create index immediately after collection creation
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
        
        # Load collection with index
        self.collection.load()
        
        # Initialize Embedding
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    def upsert_resume(self, resume_text):
        # 获取下一个可用的ID
        next_id = self.collection.num_entities

        # 文本向量化
        vector = self.embeddings.embed_query(resume_text)
        
        # 插入向量
        self.collection.insert([
            [next_id],  # 使用自增ID
            [vector],
            [resume_text]
        ])
        
        self.collection.flush()
    
    def search_resume(self, query, top_k=3):
    # 语义检索
        query_vector = self.embeddings.embed_query(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # 指定要返回的字段
        output_fields = ["text"]
        
        results = self.collection.search(
            [query_vector], 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k,
            output_fields=output_fields
        )
        
        # 处理搜索结果
        processed_results = []
        for hits in results:
            for hit in hits:
                # 直接访问实体属性
                processed_results.append({
                    'id': hit.id,
                    'distance': hit.distance,
                    'text': hit.entity.text  # 直接使用 .text 属性
                })
        
        return processed_results