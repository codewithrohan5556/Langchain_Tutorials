from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()
 
model1 = ChatGroq(model='llama-3.3-70b-versatile')

model2 = ChatGroq(model='openai/gpt-oss-120b')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate  five short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 |parser 

chain = parallel_chain | merge_chain

text = """
The Maratha Empire,[f] also referred to as the Maratha Confederacy, was an early modern state in the Indian subcontinent. For most of its existence, it comprised the realms of the Peshwa and four major independent Maratha states[14][15] under the nominal leadership of the former and nominal loyalty to the Chhatrapatis who were successors of Shivaji.

The Marathas were a Marathi-speaking warrior group from the western Deccan Plateau (present-day Maharashtra) that rose to prominence under leadership of Shivaji (17th century), who revolted against the Bijapur Sultanate and the Mughal Empire for establishing "Hindavi Swarajya" (lit.'self-rule of Hindus').[16][17] The religious attitude of Emperor Aurangzeb estranged non-Muslims, and the Maratha insurgency came at a great cost for his men and treasury.[18][19] The Maratha government also included warriors, administrators, and other nobles from other Marathi groups.[20] Shivaji's monarchy, referred to as the Maratha Kingdom,[21] expanded into a large realm in the 18th century under the leadership of Peshwa Bajirao I. Marathas from the time of Shahu I recognised the Mughal emperor as their nominal suzerain, similar to other contemporary Indian entities, though in practice, Mughal politics were largely controlled by the Marathas between 1737 and 1803.

"""
result = chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()