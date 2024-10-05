import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

# Initialize the LangChain LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key="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"
)

# Function to extract job description from URL


def extract_job_description(url):
    loader = WebBaseLoader(url)
    page_data = loader.load().pop().page_content

    prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `company`, `role`, `experience`, `skills`, and `description`.
        Only return the valid JSON.
        ### VALID JSON (NO PREAMBLE):    
        """
    )

    chain_extract = prompt_extract | llm
    res = chain_extract.invoke(input={'page_data': page_data})

    json_parser = JsonOutputParser()
    job_description = json_parser.parse(res.content)
    return job_description

# Function to generate cover letter


def generate_cover_letter(job_description, resume_content):
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### RESUME
        {resume}

        ### INSTRUCTION
        I want to apply for the above job. I have provided my resume. Generate me a cover letter for this job. Make the cover letter short with at most paragraphs. Follow below format. In this format you have to fill fields denoted with '[...]'. Skip the fields if they are not present in the resume or job_description. 
        `
            [My Name]
            [Phone Number]
            [Email Address] 
            [My Address]
            [Today's Date]

            Hiring Manager
            [Company's Name]
            [Company Address]

            Dear Sir,

            [1st Short Para]

            [2nd Short Para]

            Sincerely,
            [My Name]

            [LinkedIn Profile] | [GitHub Profile]
        `
        """
    )

    chain_email = prompt_email | llm
    res = chain_email.invoke({"job_description": str(
        job_description), "resume": resume_content})
    return res.content


# Title of the application
st.title("PDF Resume and Job Cover Letter Generator")


# Job posting URL
job_url = st.text_input("Enter Job Posting URL")
# Upload the PDF
uploaded_pdf = st.file_uploader("Choose your resume PDF file", type="pdf")

if uploaded_pdf is not None and job_url:
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_pdf.read())
        temp_file_path = temp_file.name

    # Load PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Extracted contents stored in session state
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""

    # Concatenate extracted text
    for doc in documents:
        st.session_state.extracted_text += doc.page_content + \
            "\n"  # Adding a newline for separation

    # Inform the user that the extraction is complete
    st.success("PDF text extraction complete!")

    # Extract job description from the provided URL
    job_description = extract_job_description(job_url)

    # Generate cover letter based on extracted text and job description
    cover_letter = generate_cover_letter(
        job_description, st.session_state.extracted_text)

    # Display the cover letter
    st.subheader("Generated Cover Letter")
    st.text_area("Your Cover Letter", cover_letter, height=300)

# Optionally, you can provide a button to clear the session state
if st.button("Clear Extracted Text"):
    st.session_state.extracted_text = ""
    st.write("Extracted text cleared.")
