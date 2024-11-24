import asyncio
from src.resume_rag import ResumeRAG


async def main():
    # 语义检索测试
    query = """
        1、5年及以上年开发经验；
        2、熟练掌握linux命令行操作；
        3、python/matlab代码开发经验不低于两年
        4、驾龄3年以上
        5、有软件测试经验或者有linux系统下运维经验，不低于2年
        6、有智能驾驶行业工作经验优先，有主机厂/汽车行业供应商工作经验者优先
        7、对激光雷达，毫米波雷达，超声波雷达，视觉技术有经验者优先
    """
    rag = ResumeRAG()
    results = await rag.process_query(query)
    
    # 打印结果
    print(results)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())