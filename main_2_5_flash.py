import streamlit as st
import base64
import tempfile
import os
from mistralai import Mistral
from PIL import Image
import io
from mistralai import DocumentURLChunk, ImageURLChunk
from mistralai.models import OCRResponse
import google.generativeai as genai

# Load environment variables for deployment
def load_config():
    """Load configuration from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        api_key = st.secrets.get('MISTRAL_API_KEY', '')
        google_api_key = st.secrets.get('GOOGLE_API_KEY', '')
    except:
        # Fallback to environment variables (for other platforms)
        api_key = os.getenv('MISTRAL_API_KEY', '')
        google_api_key = os.getenv('GOOGLE_API_KEY', '')
    
    return api_key, google_api_key

# Global API keys
api_key, google_api_key = load_config()

# Initialize client function with proper error handling
def initialize_mistral_client(api_key):
    """Initialize Mistral client with error handling"""
    if not api_key:
        return None
    
    try:
        client = Mistral(api_key=api_key)
        # Test the client with a simple API call
        client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Mistral client: {str(e)}")
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

# OCR Processing Functions
def upload_pdf(client, content, filename):
    """Uploads a PDF to Mistral's API and retrieves a signed URL for processing."""
    if client is None:
        raise ValueError("Mistral client is not initialized")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, filename)
        
        with open(temp_path, "wb") as tmp:
            tmp.write(content)
        
        try:
            with open(temp_path, "rb") as file_obj:
                file_upload = client.files.upload(
                    file={"file_name": filename, "content": file_obj},
                    purpose="ocr"
                )
            
            signed_url = client.files.get_signed_url(file_id=file_upload.id)
            return signed_url.url
        except Exception as e:
            raise ValueError(f"Error uploading PDF: {str(e)}")

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders with base64 encoded images in markdown."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine markdown from all pages with their respective images."""
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)

def process_ocr(client, document_source):
    """Process document with OCR API based on source type"""
    if client is None:
        raise ValueError("Mistral client is not initialized")
        
    if document_source["type"] == "document_url":
        return client.ocr.process(
            document=DocumentURLChunk(document_url=document_source["document_url"]),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
    elif document_source["type"] == "image_url":
        return client.ocr.process(
            document=ImageURLChunk(image_url=document_source["image_url"]),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
    else:
        raise ValueError(f"Unsupported document source type: {document_source['type']}")

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
            "max_output_tokens": 8192,  # Increased for Gemini 2.5 Flash
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
    global api_key, google_api_key
    
    st.set_page_config(page_title="Document OCR & Chat", layout="wide")
    
    # Sidebar: Authentication for API keys
    with st.sidebar:
        st.header("Settings")
        
        # API key inputs - only show if not already configured
        if not api_key or not google_api_key:
            st.warning("⚠️ API keys need to be configured for the app to work properly.")
            
            # API key inputs
            api_key_tab1, api_key_tab2 = st.tabs(["Mistral API", "Google API"])
            
            with api_key_tab1:
                user_api_key = st.text_input("Mistral API Key", value=api_key if api_key else "", type="password")
                if user_api_key and user_api_key != api_key:
                    api_key = user_api_key
            
            with api_key_tab2:
                user_google_api_key = st.text_input(
                    "Google API Key", 
                    value=google_api_key if google_api_key else "", 
                    type="password",
                    help="API key for Google Gemini to use for response generation"
                )
                if user_google_api_key and user_google_api_key != google_api_key:
                    google_api_key = user_google_api_key
        
        # Initialize Mistral client with the API key
        mistral_client = None
        if api_key:
            mistral_client = initialize_mistral_client(api_key)
            if mistral_client:
                st.sidebar.success("✅ Mistral API connected successfully")
        
        # Google API key validation
        if google_api_key:
            is_valid, message = test_google_api(google_api_key)
            if is_valid:
                st.sidebar.success(f"✅ Google API {message}")
            else:
                st.sidebar.error(f"❌ Google API: {message}")
                google_api_key = None
        
        # Display warnings for missing API keys
        if not api_key or mistral_client is None:
            st.sidebar.warning("⚠️ Valid Mistral API key required for document processing")
        
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
        
        # Only show document upload if Mistral client is initialized
        if mistral_client:
            input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload", "URL"])
            
            document_source = None
            
            if input_method == "URL":
                url = st.text_input("Document URL:")
                if url and st.button("Load Document from URL"):
                    document_source = {
                        "type": "document_url",
                        "document_url": url
                    }
            
            elif input_method == "PDF Upload":
                uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
                if uploaded_file and st.button("Process PDF"):
                    content = uploaded_file.read()
                    
                    try:
                        # Prepare document source for OCR processing
                        document_source = {
                            "type": "document_url",
                            "document_url": upload_pdf(mistral_client, content, uploaded_file.name)
                        }
                        
                        # Show success message and download option
                        st.success("✅ PDF uploaded and processed successfully!")
                        
                        # Provide download button for user to view the original
                        st.download_button(
                            label="📥 Download Original PDF",
                            data=content,
                            file_name=uploaded_file.name,
                            mime="application/pdf",
                            help="Download to view the original PDF file"
                        )
                        
                        st.info("💡 The document content will be extracted and available for chat once processing completes.")
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        document_source = None
            
            elif input_method == "Image Upload":
                uploaded_image = st.file_uploader("Choose Image file", type=["png", "jpg", "jpeg"])
                if uploaded_image and st.button("Process Image"):
                    try:
                        # Display the uploaded image
                        image = Image.open(uploaded_image)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        # Convert image to base64
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Prepare document source for OCR processing
                        document_source = {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_str}"
                        }
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
            
            # Process document if source is provided
            if document_source:
                with st.spinner("Processing document..."):
                    try:
                        ocr_response = process_ocr(mistral_client, document_source)
                        
                        if ocr_response and ocr_response.pages:
                            # Extract all text without page markers for clean content
                            raw_content = []
                            
                            for page in ocr_response.pages:
                                page_content = page.markdown.strip()
                                if page_content:  # Only add non-empty pages
                                    raw_content.append(page_content)
                            
                            # Join all content into one clean string for the model
                            final_content = "\n\n".join(raw_content)
                            
                            # Also create a display version with page numbers for the UI
                            display_content = []
                            for i, page in enumerate(ocr_response.pages):
                                page_content = page.markdown.strip()
                                if page_content:
                                    display_content.append(f"Page {i+1}:\n{page_content}")
                            
                            display_formatted = "\n\n----------\n\n".join(display_content)
                            
                            # Store both versions
                            st.session_state.document_content = final_content  # Clean version for the model
                            st.session_state.display_content = display_formatted  # Formatted version for display
                            st.session_state.document_loaded = True
                            
                            # Show success information about extracted content
                            st.success(f"Document processed successfully! Extracted {len(final_content)} characters from {len(raw_content)} pages.")
                        else:
                            st.warning("No content extracted from document.")
                    
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
    
    # Main area: Display chat interface
    st.title("Document OCR & Chat")
    
    # Document preview area
    if "document_loaded" in st.session_state and st.session_state.document_loaded:
        with st.expander("Document Content", expanded=False):
            # Show the display version with page numbers
            if "display_content" in st.session_state:
                st.markdown(st.session_state.display_content)
            else:
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

if __name__ == "__main__":
    main()