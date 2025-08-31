import streamlit as st
import pandas as pd
import uuid
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import groq
from key import groq_api_key

st.title("ColdMail AI Job Application Generator")

# -------------------------------
# Step 1: Get Job Page URL
# -------------------------------
job_url = st.text_input("Enter the career/job posting URL", "")

if job_url:
    loader = WebBaseLoader(job_url)
    page_data = loader.load().pop().page_content

    # -------------------------------
    # Step 2: Extract Jobs from Page
    # -------------------------------
    llm_primary = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
    )

    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTIONS:
        Extract job postings as JSON with keys: 'role', 'experience', 'skills', 'location', 'description'.
        Only return valid JSON.
        """
    )

    chain_extract = prompt_extract | llm_primary

    # -------------------------------
    # Step 3: Safe invoke function
    # -------------------------------
    def safe_invoke(chain, input_dict, primary_model="llama-3.3-70b-versatile", fallback_model="llama-3.1-8b-instant"):
        try:
            return chain.invoke(input_dict)
        except groq.RateLimitError:
            st.warning(f"Primary model '{primary_model}' rate limit reached. Switching to fallback model '{fallback_model}'...")
            fallback_llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model=fallback_model,
            )
            # Rebuild the chain
            fallback_chain = chain.steps[0] | fallback_llm
            return fallback_chain.invoke(input_dict)

    response = safe_invoke(chain_extract, {"page_data": page_data})

    json_parser = JsonOutputParser()
    job_postings = json_parser.parse(response.content)

    if not job_postings:
        st.error("No job postings found on this page.")
    else:
        job = job_postings[0]  # pick the first job for demo

        # -------------------------------
        # Step 4: Load your portfolio CSV
        # -------------------------------
        df = pd.read_csv("my_portfolio.csv")

        import chromadb
        client = chromadb.PersistentClient("vectorstore")
        collection = client.get_or_create_collection(name="portfolio")

        if not collection.count():
            for _, row in df.iterrows():
                collection.add(
                    documents=[row["TechStack"]],
                    metadatas=[{"links": row["Links"]}],
                    ids=[str(uuid.uuid4())],
                )

        query_result = collection.query(
            query_texts=job['skills'],
            n_results=2,
        ).get("metadatas", [])

        # Flatten query result
        all_links = [link for sublist in query_result for meta in sublist for link in meta.get("links", [])]
        all_techstack = [doc for sublist in collection.query(query_texts=job['skills'], n_results=2).get("documents", []) for doc in sublist]

        # -------------------------------
        # Step 5: Prepare email prompt
        # -------------------------------
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
            - Experienced in building AI-powered applications (Resume Enhancer, Suggestify, DietForge, PersonaFit)
            - Active open-source contributor (GitHub: https://github.com/AkibDa)
            - LinkedIn: https://www.linkedin.com/in/sk-akib-ahammed/

            ### PORTFOLIO MATCH:
            Tech Stack from my portfolio: {techstack}
            Relevant Project Links: {links}
            
            ### INSTRUCTIONS:
            Write a professional cold email applying for the above role.
            Return only the final email text:
            - Subject line
            - Email body
            - No preamble
            """
        )

        chain_email = prompt_email | llm_primary
        response_email = safe_invoke(chain_email, {
            "role": job['role'],
            "experience": job['experience'],
            "skills": ", ".join(job['skills']),
            "location": job['location'],
            "description": job['description'],
            "techstack": ", ".join(all_techstack),
            "links": ", ".join(all_links),
        })

        st.subheader("Generated Cold Email")
        st.text_area("Email", response_email.content, height=400)
