from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from config.config import openai_api_key
from langchain_community.vectorstores import Milvus
from core.vector_store import VectorStores
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class ResumeMatchSystem:
    def __init__(self):
        pass

    def __init__(self):
        # self.embeddings = OpenAIEmbeddings(api_key=openai_api_key) # init embedding
        self.llm = ChatOpenAI(temperature=0.1, api_key=openai_api_key) # openai chat 
        self.vector_store = VectorStores().vector_store

        # self.memory = ConversationBufferMemory(
            # memory_key="chat_history",
            # return_messages=True
        # )

    def analyze_job_resume_match(self, job_description):
        """
        分析工作岗位, 并匹配优化简历
        # 1. 检索相关度最高简历版本
        # 2. 构建简历分析prompt
        # 3. 创建分析chain
        # 4. 执行分析
        # 5. 补充需求重新执行修改建议，并生成匹配简历
        """
        # 1. 检索相关简历片段
        resume_docs = self.vector_store.similarity_search_with_score(
            job_description,
            k=2,
            # filter={'skills': 'Python'}
        )

        # for doc, score in resume_docs:
        #     print(doc)
        #     print('---' * 100)
        #     print(score)

        import pprint
        pprint.pprint(resume_docs)

        # 构建详细的匹配分析提示词模板
        prompt_template_str = """
        职位描述：{job_description}
        简历内容：{resume_text}
        
        任务：
        1. 详细分析该职位的核心技能要求
        2. 评估简历中候选人的技能和经验匹配程度
        3. 提供具体的简历优化建议
        
        输出格式：kan
        ## 职位要求分析
        - 技术栈：
        - 经验级别：
        - 关键技能：
        
        ## 简历匹配分析
        - 匹配技能：
        - 不匹配技能：
        - 经验重点：
        
        ## 简历优化建议
        1. 
        2. 
        3. 
        
        ## 匹配度评估
        匹配度：X%
        """
        
        prompt = PromptTemplate(
            input_variables=["job_description", "resume_text"], 
            template=prompt_template_str
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        # 调用 LLM 进行分析
        analysis_result = chain.invoke({
            "job_description": job_description, 
            "resume_text": resume_docs
        })

        print(analysis_result)
        
        return analysis_result



        # print(dir(self.vector_store))

        print(resume_docs)