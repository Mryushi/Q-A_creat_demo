from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from qa_model import QA
from langchain.prompts import ChatPromptTemplate
from prompt_template import system_template_text,user_template_text
import os

# 获取环境变量
api_key = os.getenv("GLM_API_KEY")

# 模型设置
model = ChatOpenAI(
    temperature=0.95,
    model="glm-4-0520",
    openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 提示模板（包括系统消息和用户消息）
prompt = ChatPromptTemplate([
    ("system", system_template_text),
    ("user", user_template_text)
])

# 输出解析器，将模型返回的内容解析成自定义格式
output_parser = PydanticOutputParser(pydantic_object=QA)

# 链式创建（通过管道将提示模板传递给模型，再传递给输出解析器，形成一个处理链）
chain = prompt | model | output_parser


# 文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    # 每小块文本的最少字数
    chunk_size=50,
    # 分割的文本间的重复字段
    chunk_overlap=0,
    # 具体分割标识符
    separators=["<end>"]
)

# 加载文档
loader = TextLoader("temp.txt", encoding='utf-8')
docs = loader.load()
# 分割文档
texts = text_splitter.split_documents(docs)

# 我未做报错处理，不会写，嘿嘿，跑的时候记得时不时看看
for text in texts:
    # 将小块文档内容作为用户输入
    question = text.page_content
    # 启动链，参数为prompt_template的变量，返回的是QA对象
    result = chain.invoke({
        "parser_instructions": output_parser.get_format_instructions(),
        "text": question
    })

    # 取出QA对象中的问题与答案，匹配并写入result.txt中
    for question, answer in zip(result.questions, result.answers):
        with open("result.txt", "a", encoding="utf-8") as result_file:
            result_file.write(f"Q: {question}\nA: {answer}\n\n")
    # 处理完一个故事，空一行作区分
    with open("result.txt", "a", encoding="utf-8") as result_file:
        result_file.write("\n")
