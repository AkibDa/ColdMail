import streamlit as st
import pandas as pd
import uuid
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import groq
from key import groq_api_key

# Page configuration
st.set_page_config(
    page_title="ColdMail Generator",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #0f1116;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #0f1116;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .stTextInput > div > div > input {
        border: 2px solid #90CAF9;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">✉️ ColdMail Generator</h1>', unsafe_allow_html=True)

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you generate personalized cold emails for job applications by:
    
    1. Extracting job details from a career page URL
    2. Matching your skills with the job requirements
    3. Generating a professional email tailored to the position
    """)
    
    st.header("Instructions")
    st.write("""
    1. Enter the URL of the job posting
    2. Wait for the AI to analyze the job requirements
    3. Review the generated email
    4. Copy and customize as needed
    """)
    
    st.header("Your Profile")
    st.write("**Name:** Sk Akib Ahammed")
    st.write("**Role:** AI Engineer and Full-Stack Developer")
    st.write("**GitHub:** https://github.com/AkibDa")
    st.write("**LinkedIn:** https://www.linkedin.com/in/sk-akib-ahammed/")

# Main content area
st.markdown('<div class="info-box">Enter the URL of the job posting you want to apply for below. The AI will analyze the job requirements and generate a personalized email for you.</div>', unsafe_allow_html=True)

# -------------------------------
# Step 1: Get Job Page URL
# -------------------------------
st.markdown('<h2 class="sub-header">Step 1: Job Information</h2>', unsafe_allow_html=True)
job_url = st.text_input("**Career/Job Posting URL**", "", placeholder="https://company.com/careers/job-id")

if job_url:
    with st.spinner("🔍 Analyzing job posting..."):
        loader = WebBaseLoader(job_url)
        page_data = loader.load().pop().page_content

    # -------------------------------
    # Step 2: Extract Jobs from Page
    # -------------------------------
    st.markdown('<h2 class="sub-header">Step 2: Job Details Extraction</h2>', unsafe_allow_html=True)
    
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
        Extract ALL job postings from the scraped text and return them as a valid JSON array.
        Each job posting should have these exact keys: 'role', 'experience', 'skills', 'location', 'description'.
        
        IMPORTANT: 
        - Return ONLY the JSON array, no additional text, explanations, or code.
        - The skills should be an array of strings.
        - If no job postings are found, return an empty array [].
        - Do not include any markdown formatting or code blocks.
        
        ### VALID JSON OUTPUT FORMAT:
        [
          {{
            "role": "Job Title",
            "experience": "Required experience",
            "skills": ["Skill1", "Skill2", "Skill3"],
            "location": "Job location",
            "description": "Job description"
          }}
        ]
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

    with st.spinner("🧠 Extracting job details..."):
        response = safe_invoke(chain_extract, {"page_data": page_data})

        json_parser = JsonOutputParser()
        job_postings = json_parser.parse(response.content)

    if not job_postings:
        st.error("No job postings found on this page.")
    else:
        job = job_postings[0]  # pick the first job for demo
        
        # Display job details in an expander
        with st.expander("View Extracted Job Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Role:**", job.get('role', 'N/A'))
                st.write("**Experience:**", job.get('experience', 'N/A'))
                st.write("**Location:**", job.get('location', 'N/A'))
            with col2:
                st.write("**Skills:**", ", ".join(job.get('skills', [])))
            
            st.write("**Description:**")
            st.write(job.get('description', 'No description available'))

        # -------------------------------
        # Step 4: Load your portfolio CSV
        # -------------------------------
        st.markdown('<h2 class="sub-header">Step 3: Skills Matching</h2>', unsafe_allow_html=True)
        
        with st.spinner("📊 Matching your skills with job requirements..."):
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
        st.markdown('<h2 class="sub-header">Step 4: Email Generation</h2>', unsafe_allow_html=True)
        
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
            """
        )

        chain_email = prompt_email | llm_primary
        
        with st.spinner("📧 Generating your personalized email..."):
            response_email = safe_invoke(chain_email, {
                "role": job['role'],
                "experience": job['experience'],
                "skills": ", ".join(job['skills']),
                "location": job['location'],
                "description": job['description'],
                "techstack": ", ".join(all_techstack),
                "links": ", ".join(all_links),
            })

        st.markdown('<div class="success-box">Your personalized cold email has been generated successfully!</div>', unsafe_allow_html=True)
        
        # Email display with copy functionality
        st.subheader("Generated Cold Email")
        email_text = st.text_area("**Email Content**", response_email.content, height=400)
        
        # Add copy button
        if st.button("📋 Copy Email to Clipboard"):
            st.write("📋 Email copied to clipboard!")
            # Note: Streamlit doesn't directly support clipboard operations in all environments
            # This is a visual indication only
            
        # Add download button
        st.download_button(
            label="💾 Download Email as Text",
            data=email_text,
            file_name=f"job_application_{job.get('role', 'position').replace(' ', '_')}.txt",
            mime="text/plain"
        )