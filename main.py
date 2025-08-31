import uuid
import pandas as pd
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from key import groq_api_key
import groq

# --- Fallback-safe invocation ---
def safe_invoke(prompt_template, input_dict,
                primary_model="llama-3.3-70b-versatile",
                fallback_model="llama-3.1-8b-instant"):

    llm_primary = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model=primary_model
    )

    chain = prompt_template | llm_primary

    try:
        return chain.invoke(input_dict)
    except groq.RateLimitError:
        print(f"Primary model '{primary_model}' rate limit reached. Switching to fallback model '{fallback_model}'...")
        llm_fallback = ChatGroq(
            temperature=0,
            groq_api_key=groq_api_key,
            model=fallback_model
        )
        fallback_chain = prompt_template | llm_fallback
        return fallback_chain.invoke(input_dict)

# --- Load job page ---
loader = WebBaseLoader("https://www.google.com/about/careers/applications/jobs/results/75808725415142086-software-engineering-intern-bs-summer-2026?q=Intern")
page_data = loader.load().pop().page_content

# --- Extract jobs from page ---
prompt_extract = PromptTemplate.from_template("""
### SCRAPED TEXT FROM WEBSITE:
{page_data}
### INSTRUCTIONS:
Extract all job postings and return them in JSON format with keys: 'role', 'experience', 'skills', 'location', 'description'.
Only return the valid JSON.
### VALID JSON (NO PREAMBLE):
""")

response = safe_invoke(prompt_extract, {"page_data": page_data})

json_parser = JsonOutputParser()
jobs = json_parser.parse(response.content)

# --- Load portfolio CSV ---
df = pd.read_csv("my_portfolio.csv")

# --- Setup Chroma vector store ---
client = chromadb.PersistentClient("vectorstore")
collection = client.get_or_create_collection(name="portfolio")

if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=[row["TechStack"]],
            metadatas=[{"links": row["Links"]}],
            ids=[str(uuid.uuid4())],
        )

# --- Pick the first job for demo ---
job = jobs[0]

# --- Query portfolio for relevant projects ---
query_result = collection.query(
    query_texts=job['skills'],
    n_results=2,
)

# Flatten Chroma results
all_docs = []
all_links = []

for item in query_result:
    if isinstance(item, dict):
        all_docs.extend(item.get("documents", []))
        all_links.extend([meta["links"] for meta in item.get("metadatas", [])])
    elif isinstance(item, list):
        for sub in item:
            if isinstance(sub, dict):
                all_docs.extend(sub.get("documents", []))
                all_links.extend([meta["links"] for meta in sub.get("metadatas", [])])

# --- Email generation prompt ---
prompt_email = PromptTemplate.from_template("""
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
- Subject line mentioning the role
- Address to Hiring Manager (use "Dear Hiring Manager," if no name)
- Intro about me (student, aspiring AI engineer, developer)
- Highlight skills/projects aligned with job
- Include one or two portfolio highlights
- Polite, confident, enthusiastic
- 150–200 words, easy to skim
- End with a call-to-action
- Sign off with full name and contact info

### OUTPUT FORMAT:
Return only the final email text:
- Subject line
- Email body (greeting, paragraphs, sign-off)
- NO PREAMBLE, no explanations, no markdown
""")

# --- Generate email ---
response_email = safe_invoke(prompt_email, {
    "role": job['role'],
    "experience": job['experience'],
    "skills": ", ".join(job['skills']),
    "location": job['location'],
    "description": job['description'],
    "techstack": ", ".join(all_docs),
    "links": ", ".join(all_links),
})

print(response_email.content)
