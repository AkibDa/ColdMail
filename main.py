from langchain_groq import ChatGroq
from key import groq_api_key

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
)


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.google.com/about/careers/applications/jobs/results/75808725415142086-software-engineering-intern-bs-summer-2026?q=Intern")
page_data = loader.load().pop().page_content
# print(page_data)

from langchain_core.prompts import PromptTemplate

prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTIONS:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing the following keys: 'role', 'experience', 'skills', 'location' and 'description'.
    Only return the valid JSON
    ### VALID JSON (NO PREAMBLE):
    """
)

chain_extract = prompt_extract | llm
response = chain_extract.invoke({"page_data": page_data})
print(response.content)
