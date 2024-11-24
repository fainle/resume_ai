from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from src.resume_vectorizer import ResumeVectorizer


class ResumeRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI()
        self.milvus_client = ResumeVectorizer()  # 自定义的 Milvus 客户端
    
    async def process_query(self, job_description: str, top_k: int = 3):
        # 1. 查询处理
        processed_query = self.preprocess_query(job_description)
        
        print(type(job_description))

        # 2. 向量化
        # query_embedding = await self.embeddings.aembed_query(processed_query)
        
        # print(query_embedding)

        # 3. 相似度检索
        similar_resumes = self.milvus_client.search_similar_resumes(processed_query, top_k=1)
        
        # 4. 上下文组装
        context = self.build_context(similar_resumes)
        
        # 5. 提示构建
        prompt = f"""
        基于以下简历信息和职位要求，分析匹配度并给出建议：
        
        职位要求：{job_description}
        
        相关简历信息：{context}
        
        请提供：
        1. 整体匹配度分析
        2. 技能匹配详情
        3. 改进建议
        """
        
        # 6. LLM 生成
        response = await self.llm.agenerate([prompt])
        print(response)
        
        return {
            "analysis": response.generations[0][0].text,
            "similar_resumes": similar_resumes,
            # "match_score": self.calculate_match_score(similar_resumes[0])
        }

    def preprocess_query(self, query: str) -> str:
        """查询预处理，提取关键信息"""
        processed = query.lower()
        # 提取关键技能、要求等
        return processed
    
    def build_context(self, similar_resumes: list) -> str:
        """构建上下文信息"""
        context = []
        for resume in similar_resumes:
            context.append(f"""
            技能: {resume}
            经验: {resume}
            项目: {resume}
            """)
        return "\n".join(context)
    
    # def calculate_match_score(self, resume) -> float:
    #     """计算匹配分数"""
    #     # 实现匹配度计算逻辑
    #     return score