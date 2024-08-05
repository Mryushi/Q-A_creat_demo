from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


# 定义输出解析器返回对象
class QA(BaseModel):
    questions: List[str] = Field(description="根据用户提供的文本块生成一些有意义的问题")
    answers: List[str] = Field(description="每个问题的答案")
