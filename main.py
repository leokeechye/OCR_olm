import streamlit as st
import base64
import tempfile
import os
import json
from PIL import Image
import io
import google.generativeai as genai
import replicate
import requests
from io import BytesIO
import fitz  # PyMuPDF for PDF processing

# Load environment variables for deployment
def load_config():
    """Load configuration from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        replicate_api_key = st.secrets.get('REPLICATE_API_TOKEN', '')
        google_api_key = st.secrets.get('GOOGLE_API_KEY', '')
    except:
        # Fallback to environment variables (for other platforms)
        replicate_api_key = os.getenv('REPLICATE_API_TOKEN', '')
        google_api_key = os.getenv('GOOGLE_API_KEY', '')
    
    return replicate_api_key, google_api_key

# Global API keys
replicate_api_key, google_api_key = load_config()

# Initialize Replicate client function with proper error handling
def initialize_replicate_client(api_key):
    """Initialize Replicate client with error handling"""
    if not api_key:
        return None
    
    try:
        # Set the API token
        os.environ["REPLICATE_API_TOKEN"] = api_key
        client = replicate.Client(api_token=api_key)
        
        # Test the client with a simple API call
        list(client.models.list())
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Replicate client: {str(e)}")
        return None

# Test Google API key
def test_google_api(api_key):
    """Test Google API key and return status"""
    if not api_key:
        return False, "No API key provided"
    
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return True, "Connected successfully"
    except Exception as e:
        return False, f"Failed to connect: {str(e)}"

# File upload helper functions
def upload_file_to_temporary_url(file_content, filename):
    """Upload file to a temporary hosting service and return URL"""
    # For this example, we'll save to temp file and use a file upload service
    # In production, you might want to use a proper file hosting service
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name
    
    try:
        # For demo purposes, we'll use the file directly
        # In production, upload to a service like AWS S3, Google Cloud Storage, etc.
        return temp_path, temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def cleanup_temp_file(file_path):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except:
        pass

# PDF to Image conversion
def pdf_to_image(pdf_content, page_number=1, max_dim=1024):
    """Convert PDF page to image with specified max dimension"""
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Check if page exists
        if page_number > len(pdf_document):
            raise ValueError(f"Page {page_number} does not exist. PDF has {len(pdf_document)} pages.")
        
        # Get the page (0-indexed)
        page = pdf_document.load_page(page_number - 1)
        
        # Calculate zoom factor to make longest dimension = max_dim
        rect = page.rect
        zoom_x = max_dim / max(rect.width, rect.height)
        zoom_y = zoom_x
        
        # Create transformation matrix
        mat = fitz.Matrix(zoom_x, zoom_y)
        
        # Render page to image
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        
        pdf_document.close()
        
        return image
        
    except Exception as e:
        raise ValueError(f"Error converting PDF to image: {str(e)}")

# OCR Processing Functions using Qwen2-VL (base model for OlmoCR)
def process_pdf_with_olmocr(client, pdf_content, filename, page_number=1):
    """Process PDF using Qwen2-VL (base model of OlmoCR)"""
    if client is None:
        raise ValueError("Replicate client is not initialized")
    
    try:
        # Convert PDF page to image
        image = pdf_to_image(pdf_content, page_number, max_dim=1024)
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_url = f"data:image/png;base64,{img_base64}"
        
        # Use Qwen2-VL model with OCR-focused prompt
        # This is the base model that OlmoCR is fine-tuned from
        output = client.run(
            "lucataco/qwen2-vl-7b-instruct",
            input={
                "image": image_url,
                "prompt": """Please extract all text from this document image. 
                
Follow these instructions:
1. Read the text in natural reading order (left to right, top to bottom)
2. Preserve the structure and formatting as much as possible
3. Include all visible text, headers, paragraphs, tables, captions, etc.
4. If there are multiple columns, read them in the correct order
5. Maintain line breaks and spacing where appropriate
6. Output the text in a clean, readable format

Extract all text:""",
                "max_tokens": 4096
            }
        )
        
        return output
        
    except Exception as e:
        # Fallback: try the original olmocr model with simpler parameters
        try:
            output = client.run(
                "lucataco/olmocr-7b",
                input={
                    "image": image_url,
                    "prompt": "Extract all text from this document in natural reading order."
                }
            )
            return output
        except Exception as e2:
            raise ValueError(f"Error processing PDF. Qwen2-VL attempt: {str(e)}. OlmoCR attempt: {str(e2)}")

def process_image_with_olmocr(image_data):
    """Process image using Google Gemini Vision (since OlmoCR is primarily for PDFs)"""
    try:
        genai.configure(api_key=google_api_key)
        
        # Convert image data to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Use Gemini Vision for image OCR
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Extract all text from this image in a structured format. 
        Preserve the layout and formatting as much as possible. 
        Return the text in markdown format."""
        
        response = model.generate_content([prompt, image])
        
        return response.text
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def parse_olmocr_output(output):
    """Parse the output from OlmoCR model"""
    try:
        # The output should be a string containing the extracted text
        if isinstance(output, str):
            return output.strip()
        else:
            return str(output).strip()
    except Exception as e:
        return f"Error parsing output: {str(e)}"

def generate_response(context, query):
    """Generate a response using Google Gemini API"""
    try:
        # Initialize the Google Gemini API
        genai.configure(api_key=google_api_key)
        
        # Check for empty context
        if not context or len(context) < 10:
            return "Error: No document content available to answer your question."
            
        # Create a prompt with the document content and query
        prompt = f"""I have a document with the following content:

{context}

Based on this document, please answer the following question:
{query}

If you can find information related to the query in the document, please answer based on that information.
If the document doesn't specifically mention the exact information asked, please try to infer from related content or clearly state that the specific information isn't available in the document.
"""
        
        # Generate response using Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
def main():
    global replicate_api_key, google_api_key
    
    st.set_page_config(page_title="Document OCR & Chat with OlmoCR", layout="wide")
    
    # Sidebar: Authentication for API keys
    with st.sidebar:
        st.header("Settings")
        
        # API key inputs - only show if not already configured
        if not replicate_api_key or not google_api_key:
            st.warning("⚠️ API keys need to be configured for the app to work properly.")
            
            # API key inputs
            api_key_tab1, api_key_tab2 = st.tabs(["Replicate API", "Google API"])
            
            with api_key_tab1:
                user_replicate_key = st.text_input("Replicate API Token", value=replicate_api_key if replicate_api_key else "", type="password")
                if user_replicate_key and user_replicate_key != replicate_api_key:
                    replicate_api_key = user_replicate_key
            
            with api_key_tab2:
                user_google_api_key = st.text_input(
                    "Google API Key", 
                    value=google_api_key if google_api_key else "", 
                    type="password",
                    help="API key for Google Gemini to use for response generation"
                )
                if user_google_api_key and user_google_api_key != google_api_key:
                    google_api_key = user_google_api_key
        
        # Initialize Replicate client with the API key
        replicate_client = None
        if replicate_api_key:
            replicate_client = initialize_replicate_client(replicate_api_key)
            if replicate_client:
                st.sidebar.success("✅ Replicate API connected successfully")
        
        # Google API key validation
        if google_api_key:
            is_valid, message = test_google_api(google_api_key)
            if is_valid:
                st.sidebar.success(f"✅ Google API {message}")
            else:
                st.sidebar.error(f"❌ Google API: {message}")
                google_api_key = None
        
        # Display warnings for missing API keys
        if not replicate_api_key or replicate_client is None:
            st.sidebar.warning("⚠️ Valid Replicate API token required for document processing")
        
        if not google_api_key:
            st.sidebar.warning("⚠️ Google API key required for chat functionality")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "document_content" not in st.session_state:
            st.session_state.document_content = ""
        
        if "document_loaded" not in st.session_state:
            st.session_state.document_loaded = False
        
        # Document upload section
        st.subheader("Document Upload")
        
        # Only show document upload if Replicate client is initialized
        if replicate_client:
            input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload"])
            
            if input_method == "PDF Upload":
                uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
                
                # Page selection for multi-page PDFs
                page_number = st.number_input("Page Number", min_value=1, value=1, help="Which page to extract (starting from 1)")
                
                if uploaded_file and st.button("Process PDF"):
                    content = uploaded_file.read()
                    
                    try:
                        # First, check how many pages the PDF has
                        try:
                            pdf_doc = fitz.open(stream=content, filetype="pdf")
                            total_pages = len(pdf_doc)
                            pdf_doc.close()
                            
                            if page_number > total_pages:
                                st.error(f"Page {page_number} does not exist. This PDF has {total_pages} pages.")
                                return
                                
                            st.info(f"PDF has {total_pages} pages. Processing page {page_number}...")
                        except Exception as e:
                            st.error(f"Error reading PDF: {str(e)}")
                            return
                        
                        with st.spinner(f"Converting page {page_number} to image and processing with Qwen2-VL/OlmoCR..."):
                            # Show the converted image for reference
                            try:
                                preview_image = pdf_to_image(content, page_number, max_dim=1024)
                                st.image(preview_image, caption=f"Page {page_number} (processed image)", width=400)
                            except Exception as e:
                                st.warning(f"Could not display preview: {str(e)}")
                            
                            # Process with Qwen2-VL/OlmoCR
                            ocr_output = process_pdf_with_olmocr(
                                replicate_client, 
                                content, 
                                uploaded_file.name, 
                                page_number
                            )
                            
                            # Parse the output
                            extracted_text = parse_olmocr_output(ocr_output)
                            
                            if extracted_text and len(extracted_text.strip()) > 0:
                                st.session_state.document_content = extracted_text
                                st.session_state.document_loaded = True
                                
                                st.success(f"✅ PDF page {page_number} processed successfully! Extracted {len(extracted_text)} characters.")
                                
                                # Provide download button for user to view the original
                                st.download_button(
                                    label="📥 Download Original PDF",
                                    data=content,
                                    file_name=uploaded_file.name,
                                    mime="application/pdf",
                                    help="Download to view the original PDF file"
                                )
                            else:
                                st.warning("No content extracted from the PDF page.")
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
            
            elif input_method == "Image Upload":
                uploaded_image = st.file_uploader("Choose Image file", type=["png", "jpg", "jpeg"])
                if uploaded_image and st.button("Process Image"):
                    try:
                        # Display the uploaded image
                        image = Image.open(uploaded_image)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        with st.spinner("Processing image with Gemini Vision..."):
                            # Process image with Gemini Vision
                            extracted_text = process_image_with_olmocr(uploaded_image.getvalue())
                            
                            if extracted_text and len(extracted_text.strip()) > 0:
                                st.session_state.document_content = extracted_text
                                st.session_state.document_loaded = True
                                st.success(f"✅ Image processed successfully! Extracted {len(extracted_text)} characters.")
                            else:
                                st.warning("No content extracted from the image.")
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
        else:
            st.info("Please configure your Replicate API token to upload documents.")
    
    # Main area: Display chat interface
    st.title("Document OCR & Chat with Qwen2-VL / OlmoCR")
    
    # Document preview area
    if "document_loaded" in st.session_state and st.session_state.document_loaded:
        with st.expander("Extracted Document Content", expanded=False):
            st.markdown(st.session_state.document_content)
        
        # Chat interface
        st.subheader("Chat with your document")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input for user query
        if prompt := st.chat_input("Ask a question about your document..."):
            # Check if Google API key is available
            if not google_api_key:
                st.error("Google API key is required for generating responses. Please add it in the sidebar settings.")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Show thinking spinner
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Get document content from session state
                        document_content = st.session_state.document_content
                        
                        # Generate response directly
                        response = generate_response(document_content, prompt)
                        
                        # Display response
                        st.markdown(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Show a welcome message if no document is loaded
        st.info("👈 Please upload a document using the sidebar to start chatting.")
        
        # Show model information
        st.markdown("""
        ### About this app
        
        This application uses:
        - **Qwen2-VL-7B**: Primary OCR model (base model for OlmoCR) for PDF processing
        - **OlmoCR-7B**: Fallback OCR model (if available) for PDF processing
        - **Google Gemini Vision**: Alternative for image OCR processing  
        - **Google Gemini 2.5 Flash**: For chat functionality
        
        Qwen2-VL-7B is a powerful vision-language model that excels at extracting text from document images.
        OlmoCR-7B is a specialized fine-tuned version of Qwen2-VL specifically optimized for OCR tasks.
        """)

if __name__ == "__main__":
    main()