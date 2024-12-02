from langchain_milvus import Milvus
from langchain_openai.embeddings import OpenAIEmbeddings
from config.config import openai_api_key
from pymilvus import connections, utility, FieldSchema, CollectionSchema, Collection, DataType



class VectorStores:
    def __init__(self, collection_name="resumes", embedding_dim=1536):
        self.embedding_dim = embedding_dim

         # 连接 Milvus
        connections.connect(alias="default", host='localhost', port='19530')

        # 检查集合是否存在，如果不存在则创建
        if not utility.has_collection(collection_name):
            self._create_collection(collection_name)

        self.vector_store = Milvus(
            collection_name=collection_name,
            vector_field='resume_vector',
            text_field='content', 
            embedding_function=OpenAIEmbeddings(api_key=openai_api_key),
            connection_args={"uri": 'http://localhost:19530'},
        )

    def _create_collection(self, collection_name):
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
        self.collection = Collection(name=collection_name, schema=schema)

        #创建索引
        index_params = {
            "metric_type": "L2",
            'index_type': "IVF_FLAT",
            "params": {'nlist': 1024},
        }

        self.collection.create_index(field_name='resume_vector', index_params=index_params)
