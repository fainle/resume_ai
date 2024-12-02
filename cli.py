import click
import openai

from core.resume_match import ResumeMatchSystem

# 设置 OpenAI API 密钥
openai.api_key = "your_openai_api_key"
# 保存对话历史的列表
conversation_history = []


@click.group()
def cli():
    """一个支持多命令的终端工具，用于与 ChatGPT 交互。"""
    pass


@cli.command()
def resume_match():
    """
    一个基于终端的多轮对话工具
    """
    click.echo("欢迎使用resume rag系统, 输入JD进行简历匹配和分析, 输入 'exit' 或 'quit' 退出对话。")
    
    while True:
        # 获取用户输入
        user_input = click.prompt("输入")
        
        # 退出条件
        if user_input.lower() in ["exit", "quit"]:
            click.echo("感谢使用，再见！")
            break

        res = ResumeMatchSystem().analyze_job_resume_match(user_input)

        # # 将用户输入添加到对话历史
        # conversation_history.append({"role": "user", "content": user_input})

        # try:
        #     # 调用 OpenAI 接口
        #     response = openai.ChatCompletion.create(
        #         model=model,
        #         messages=conversation_history
        #     )
            
        #     # 获取模型回复
        #     assistant_reply = response["choices"][0]["message"]["content"]
        #     click.echo(f"ChatGPT: {assistant_reply}")
            
        #     # 将模型回复添加到对话历史
        #     conversation_history.append({"role": "assistant", "content": assistant_reply})
        # except Exception as e:
        #     click.echo(f"发生错误: {e}")


@cli.command()
def resume_save(question, model):
    pass


if __name__ == "__main__":
    cli()

# from src.resume_processor import ResumeProcessor
# from src.vector_store import ResumeVectorStore
# from src.resume_vectorizer import ResumeVectorizer

# def main():
#     # 处理简历
#     processor = ResumeProcessor('./data/resume.json')
#     resume_data = processor.parse_resume()
    
#     # 初始化向量存储
#     resume_vectorizer = ResumeVectorizer()
    
#     # 存储简历
#     resume_vectorizer.store_resume(str(resume_data), resume_data)
    

# if __name__ == "__main__":
#     main()