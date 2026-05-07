from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence,RunnablePassthrough,RunnableBranch

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)
# report_gen_chain = prompt1 | model | parser


branch_chain = RunnableBranch(
    (lambda x:len(x.split()>300),RunnableSequence(prompt2,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukriane'}))

