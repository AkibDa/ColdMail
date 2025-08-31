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
# print(response.content)

from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()
json_response = json_parser.parse(response.content)
# print(json_response)

import pandas as pd
df =  pd.read_csv("my_portfolio.csv")
# print(df.head())

import chromadb, uuid

client = chromadb.PersistentClient("vectorstore")
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=row["TechStack"],
            metadatas = {"links": row["Links"]},
            ids=[str(uuid.uuid4())],
        )
  
job = json_response[0]  
        
query_result = collection.query(
    query_texts=job['skills'],
    n_results=2,
).get("metadatas", [])
# print(links)

if isinstance(query_result, list):
    query_result = query_result[0]

docs = query_result.get("documents", [])
metas = query_result.get("metadatas", [])

all_links = [meta["links"] for sublist in metas for meta in sublist]
all_techstack = [doc for sublist in docs for doc in sublist]

prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    Role: {role}
    Experience: {experience}
    Skills: {skills}
    Location: {location}
    Description: {description}
    
    ### ABOUT ME:
    - Name: Sk Akib Ahammed
    - Aspiring AI Engineer and Full-Stack Developer
    - Strong foundation in Python, Java, JavaScript, HTML, CSS, and C
    - Experienced in building AI-powered applications (e.g., Resume Enhancer, Suggestify, DietForge, PersonaFit)
    - Active open-source contributor (GitHub: https://github.com/AkibDa)
    - LinkedIn: https://www.linkedin.com/in/sk-akib-ahammed/
    - Passionate about solving real-world problems with AI and software engineering

    ### PORTFOLIO MATCH:
    Tech Stack from my portfolio: {techstack}
    Relevant Project Links: {links}
    
    ### INSTRUCTIONS:
    Write a professional cold email applying for the above role:
    - Use a clear subject line mentioning the role (e.g., "Application for {role}").
    - Address it to the Hiring Manager (use "Dear Hiring Manager," if no name).
    - Start with a brief intro about me (student, aspiring AI engineer, developer).
    - Highlight how my skills + projects align with the job requirements.
    - Mention one or two portfolio highlights (from techstack/links).
    - Keep it polite, confident, and enthusiastic (not robotic).
    - Length: 150–200 words, easy to skim.
    - End with a call-to-action (e.g., "I’d love the chance to discuss how I can contribute to your team").
    - Sign off with my full name and contact info.

    ### OUTPUT FORMAT:
    Return only the final email text:
    - Subject line
    - Email body (with greeting, paragraphs, sign-off)
    - NO PREAMBLE, no explanations, no markdown formatting
    """
)

chain_email = prompt_email | llm
response_email = chain_email.invoke({
    "role": job['role'],
    "experience": job['experience'],
    "skills": ", ".join(job['skills']),
    "location": job['location'],
    "description": job['description'],
    "techstack": ", ".join(all_techstack),
    "links": ", ".join(all_links),
})
print(response_email.content)