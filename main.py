from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.google.com/about/careers/applications/jobs/results/75808725415142086-software-engineering-intern-bs-summer-2026?q=Intern")
page_data = loader.load().pop().page_content
print(page_data)