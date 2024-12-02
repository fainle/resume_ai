import asyncio
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


class ResumeMatchSystem:
    def __init__(self):
        # self.embeddings = OpenAIEmbeddings() # init embedding
        # self.llm = ChatOpenAI(temperature=0.7) # openai chat 

        # self.vector_store = self._init_vector_store()
        # self.memory = ConversationBufferMemory(
            # memory_key="chat_history",
            # return_messages=True
        # )
        pass

    def analyze_job_resume_match(self, job_description):
        """
        分析工作岗位, 并匹配优化简历
        # 1. 检索相关度最高简历版本
        # 2. 构建简历分析prompt
        # 3. 创建分析chain
        # 4. 执行分析
        # 5. 补充需求重新执行修改建议，并生成匹配简历
        """
        print(job_description)

if __name__ == "__main__":
    job_description = """
        职位描述：开发和维护公司的软件产品。
        我们希望你：
        1、本科及以上学历，计算机相关专业毕业2年以上。
        2、2年以上的python开发经验，熟悉python的web框架，Tornado、Django、Flask及爬虫框架scrapy，有扎实的编程功底，热爱编程
        3、熟悉Linux环境下开发及部署。
        4、熟练掌握Python数据类型，对数据结构、算法有一定的理解。
        5、熟练掌握PostgreSQL\mysql、redis数据库。
        岗位职责：
        1.配合水工艺产品经理对现有高级算法进行迭代实现以及新算法实现；
        2.搭建python高效框架，开发边缘控制器产品；
        3.熟悉PostgreSQL数据库，能熟练编写SQL语句；
        4.熟练掌握Git代码管理工具以及各项命令；
        5.熟练掌握Dockers自动运维部署工具平台。
        工作要求：
        1.两年以上Python web开发经验，熟练Python编程；
        2.熟练使用python库，熟练Python的高效写法；
        3.熟练掌握python web框架，如Django，Flask；
        4.具备水务相关项目经验优先；
        5.有边缘端开发经验优先；
        6.有意向在工业互联网方向长期发展。
    """
    ResumeMatchSystem().analyze_job_resume_match(job_description)