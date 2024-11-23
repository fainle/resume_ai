from src.resume_processor import ResumeProcessor
from src.vector_store import ResumeVectorStore
from src.resume_vectorizer import ResumeVectorizer

def main():
    # 处理简历
    processor = ResumeProcessor('./data/resume.json')
    resume_data = processor.parse_resume()
    
    # 初始化向量存储
    resume_vectorizer = ResumeVectorizer()
    
    # 存储简历
    resume_vectorizer.store_resume(str(resume_data), resume_data)
    
    # 语义检索测试
    query = "Python开发经验"
    results = resume_vectorizer.search_similar_resumes(query)
    
    # 打印结果
    print(results)

if __name__ == "__main__":
    main()