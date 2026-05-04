from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"  
)

class Person(BaseModel):
    name : str = Field(description='name of the person')
    age : int = Field(gt=18,description='age of the person')
    city : str = Field(description='name of the city the person belongs to')


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain = template | model | parser
final_result = chain.invoke({'place':'indian'})
# prompt = template.invoke({'place':'american'})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
print(final_result)
