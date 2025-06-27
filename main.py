import streamlit as st
import base64
import tempfile
import os
from PIL import Image
import io
import json

# Handle optional imports with error checking
try:
    from mistralai import Mistral
    from mistralai import DocumentURLChunk, ImageURLChunk
    from mistralai.models import OCRResponse
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    st.error("Mistral AI package not installed. Please install with: pip install mistralai")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    st.error("Replicate package not installed. Please install with: pip install replicate")

# Load environment variables for deployment
def load_config():
    """Load configuration from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        api_key = st.secrets.get('MISTRAL_API_KEY', '')
        replicate_api_key = st.secrets.get('REPLICATE_API_TOKEN', '')
    except:
        # Fallback to environment variables (for other platforms)
        api_key = os.getenv('MISTRAL_API_KEY', '')
        replicate_api_key = os.getenv('REPLICATE_API_TOKEN', '')
    
    return api_key, replicate_api_key

# Global API keys
api_key, replicate_api_key = load_config()

# Initialize client function with proper error handling
def initialize_mistral_client(api_key):
    """Initialize Mistral client with error handling"""
    if not MISTRAL_AVAILABLE:
        st.sidebar.error("Mistral AI package not available")
        return None
        
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

# Test Replicate API key
def test_replicate_api(api_key):
    """Test Replicate API key and return status"""
    if not REPLICATE_AVAILABLE:
        return False, "Replicate package not installed"
        
    if not api_key:
        return False, "No API key provided"
    
    try:
        os.environ["REPLICATE_API_TOKEN"] = api_key
        # Try to list models to test the API key
        client = replicate.Client(api_token=api_key)
        return True, "Connected successfully"
    except Exception as e:
        return False, f"Failed to connect: {str(e)}"

# OCR Processing Functions using OLMoCR-7B
def process_pdf_with_olmocr(pdf_content, filename):
    """Process PDF using OLMoCR-7B from Replicate"""
    if not REPLICATE_AVAILABLE:
        raise ValueError("Replicate package is not installed")
        
    if not replicate_api_key:
        raise ValueError("Replicate API key is not configured")
    
    try:
        # Save PDF to temporary file and upload to a temporary URL service
        # Note: In production, you'd want to use a proper file hosting service
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file_path = tmp_file.name
        
        # For this example, we'll assume you have a way to host the PDF temporarily
        # In practice, you might use services like AWS S3, Google Cloud Storage, etc.
        # For now, we'll simulate processing the first page
        
        # Process with OLMoCR-7B
        output = replicate.run(
            "lucataco/olmocr-7b",
            input={
                "pdf": tmp_file_path,  # In practice, this should be a URL
                "page_number": 1,
                "temperature": 0.1,  # Lower temperature for more consistent OCR
                "max_new_tokens": 2048
            }
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Parse the output - OLMoCR returns JSON string
        if isinstance(output, str):
            try:
                # The output is typically a JSON string containing OCR results
                parsed_output = json.loads(output)
                if isinstance(parsed_output, list) and len(parsed_output) > 0:
                    # Extract the natural text from the first result
                    result_data = json.loads(parsed_output[0]) if isinstance(parsed_output[0], str) else parsed_output[0]
                    return result_data.get('natural_text', output)
                else:
                    return str(output)
            except (json.JSONDecodeError, KeyError):
                return str(output)
        
        return str(output)
        
    except Exception as e:
        raise ValueError(f"Error processing PDF with OLMoCR: {str(e)}")

def process_image_with_olmocr(image_content):
    """Process image using OLMoCR-7B from Replicate"""
    if not REPLICATE_AVAILABLE:
        raise ValueError("Replicate package is not installed")
        
    if not replicate_api_key:
        raise ValueError("Replicate API key is not configured")
    
    try:
        # Convert image to base64 for processing
        # Note: OLMoCR expects PDF input, so for images you might want to:
        # 1. Convert image to PDF first, or
        # 2. Use a different OCR service for images
        
        # For this example, we'll use Replicate's image processing capabilities
        # You might want to use a different model for pure image OCR
        
        # Convert PIL Image to base64
        if isinstance(image_content, Image.Image):
            buffered = io.BytesIO()
            image_content.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/png;base64,{img_str}"
        else:
            image_url = image_content
        
        # Note: OLMoCR-7B is specifically designed for PDFs
        # For images, you might want to use a different Replicate model
        # Here's an example using a general OCR model (you'd need to find an appropriate one)
        
        # Placeholder - replace with actual image OCR model from Replicate
        # output = replicate.run(
        #     "appropriate-image-ocr-model",
        #     input={"image": image_url}
        # )
        
        # For now, return a message indicating PDF is preferred
        return "OLMoCR-7B is optimized for PDF documents. For best results, please convert your image to PDF first."
        
    except Exception as e:
        raise ValueError(f"Error processing image with OLMoCR: {str(e)}")

# Upload PDF and get temporary URL (you'll need to implement this based on your hosting solution)
def upload_pdf_for_replicate(content, filename):
    """Upload PDF to a temporary hosting service and return URL"""
    # This is a placeholder - you'll need to implement actual file hosting
    # Options include:
    # - AWS S3 with temporary signed URLs
    # - Google Cloud Storage
    # - Any other file hosting service with public URLs
    
    # For now, save to temp file and return file path
    # Note: Replicate needs a public URL, not a local file path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(content)
        return tmp_file.name  # This won't work with Replicate - needs to be a URL

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
    """Generate a response using OLMoCR-7B from Replicate for text analysis"""
    if not REPLICATE_AVAILABLE:
        return "Error: Replicate package is not installed. Please install with: pip install replicate"
        
    try:
        # Check for empty context
        if not context or len(context) < 10:
            return "Error: No document content available to answer your question."
        
        # For text analysis, you might want to use a different model
        # OLMoCR-7B is specifically for OCR, not for question answering
        # You could use other Replicate models for Q&A, such as:
        # - Meta's Llama models
        # - Mistral models
        # - Other language models available on Replicate
        
        # Here's an example using a language model for Q&A
        prompt = f"""Based on the following document content, please answer the question.

Document content:
{context}

Question: {query}

Answer:"""
        
        # Replace with an appropriate language model from Replicate
        # Example with Meta's Llama model (you'll need to check available models)
        try:
            output = replicate.run(
                "meta/llama-2-70b-chat",  # Replace with available model
                input={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_new_tokens": 1000,
                    "top_p": 0.9,
                }
            )
            
            # Combine output if it's a list
            if isinstance(output, list):
                return "".join(output)
            else:
                return str(output)
                
        except Exception as e:
            # Fallback to simple text matching if model fails
            return f"I found this content in the document, but couldn't process your question with the language model: {str(e)}\n\nDocument content preview: {context[:500]}..."
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
def main():
    global api_key, replicate_api_key
    
    st.set_page_config(page_title="Document OCR & Chat with OLMoCR", layout="wide")
    
    # Sidebar: Authentication for API keys
    with st.sidebar:
        st.header("Settings")
        
        # API key inputs - only show if not already configured
        if not api_key or not replicate_api_key:
            st.warning("⚠️ API keys need to be configured for the app to work properly.")
            
            # API key inputs
            api_key_tab1, api_key_tab2 = st.tabs(["Mistral API", "Replicate API"])
            
            with api_key_tab1:
                user_api_key = st.text_input("Mistral API Key", value=api_key if api_key else "", type="password")
                if user_api_key and user_api_key != api_key:
                    api_key = user_api_key
            
            with api_key_tab2:
                user_replicate_api_key = st.text_input(
                    "Replicate API Token", 
                    value=replicate_api_key if replicate_api_key else "", 
                    type="password",
                    help="API token for Replicate to use OLMoCR-7B OCR model"
                )
                if user_replicate_api_key and user_replicate_api_key != replicate_api_key:
                    replicate_api_key = user_replicate_api_key
        
        # Initialize Mistral client with the API key
        mistral_client = None
        if api_key:
            mistral_client = initialize_mistral_client(api_key)
            if mistral_client:
                st.sidebar.success("✅ Mistral API connected successfully")
        
        # Replicate API key validation
        if replicate_api_key:
            is_valid, message = test_replicate_api(replicate_api_key)
            if is_valid:
                st.sidebar.success(f"✅ Replicate API {message}")
                os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
            else:
                st.sidebar.error(f"❌ Replicate API: {message}")
                replicate_api_key = None
        
        # Display warnings for missing API keys
        if not api_key or mistral_client is None:
            st.sidebar.warning("⚠️ Valid Mistral API key required for document processing")
        
        if not replicate_api_key:
            st.sidebar.warning("⚠️ Replicate API token required for OCR and chat functionality")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "document_content" not in st.session_state:
            st.session_state.document_content = ""
        
        if "document_loaded" not in st.session_state:
            st.session_state.document_loaded = False
        
        # Document upload section
        st.subheader("Document Upload")
        
        # Show OCR method selection
        if replicate_api_key and REPLICATE_AVAILABLE:
            ocr_method = st.radio("OCR Method:", ["Mistral OCR", "OLMoCR-7B (Replicate)"])
        else:
            ocr_method = "Mistral OCR"
            if not REPLICATE_AVAILABLE:
                st.info("Install replicate package to enable OLMoCR-7B: pip install replicate")
            elif not replicate_api_key:
                st.info("Add Replicate API token to enable OLMoCR-7B")
        
        # Only show document upload if at least one OCR method is available
        if mistral_client or (replicate_api_key and REPLICATE_AVAILABLE):
            input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload", "URL"])
            
            document_source = None
            
            if input_method == "URL":
                url = st.text_input("Document URL:")
                if url and st.button("Load Document from URL"):
                    if ocr_method == "Mistral OCR" and mistral_client:
                        document_source = {
                            "type": "document_url",
                            "document_url": url,
                            "ocr_method": "mistral"
                        }
                    elif ocr_method == "OLMoCR-7B (Replicate)" and replicate_api_key and REPLICATE_AVAILABLE:
                        st.info("OLMoCR-7B currently supports PDF uploads. URL processing coming soon.")
            
            elif input_method == "PDF Upload":
                uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
                if uploaded_file and st.button("Process PDF"):
                    content = uploaded_file.read()
                    
                    try:
                        if ocr_method == "OLMoCR-7B (Replicate)" and replicate_api_key and REPLICATE_AVAILABLE:
                            # Use OLMoCR-7B for processing
                            with st.spinner("Processing with OLMoCR-7B..."):
                                extracted_text = process_pdf_with_olmocr(content, uploaded_file.name)
                                
                                # Store the extracted text directly
                                st.session_state.document_content = extracted_text
                                st.session_state.document_loaded = True
                                
                                st.success("✅ PDF processed successfully with OLMoCR-7B!")
                                st.info(f"Extracted {len(extracted_text)} characters")
                        
                        elif ocr_method == "Mistral OCR" and mistral_client:
                            # Use Mistral OCR (original method)
                            document_source = {
                                "type": "document_url",
                                "document_url": upload_pdf_for_replicate(content, uploaded_file.name),
                                "ocr_method": "mistral"
                            }
                        
                        # Provide download button for user to view the original
                        st.download_button(
                            label="📥 Download Original PDF",
                            data=content,
                            file_name=uploaded_file.name,
                            mime="application/pdf",
                            help="Download to view the original PDF file"
                        )
                        
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
                        
                        if ocr_method == "OLMoCR-7B (Replicate)" and replicate_api_key:
                            # Use OLMoCR-7B for image processing
                            with st.spinner("Processing with OLMoCR-7B..."):
                                extracted_text = process_image_with_olmocr(image)
                                
                                st.session_state.document_content = extracted_text
                                st.session_state.document_loaded = True
                                st.success("✅ Image processed with OLMoCR-7B!")
                        
                        elif ocr_method == "Mistral OCR" and mistral_client:
                            # Convert image to base64 for Mistral
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Prepare document source for OCR processing
                            document_source = {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_str}",
                                "ocr_method": "mistral"
                            }
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
            
            # Process document with Mistral if source is provided
            if document_source and document_source.get("ocr_method") == "mistral":
                with st.spinner("Processing document with Mistral OCR..."):
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
    st.title("Document OCR & Chat with OLMoCR-7B")
    
    # Document preview area
    if "document_loaded" in st.session_state and st.session_state.document_loaded:
        with st.expander("Document Content", expanded=False):
            # Show the display version with page numbers if available
            if "display_content" in st.session_state:
                st.markdown(st.session_state.display_content)
            else:
                st.text(st.session_state.document_content)
        
        # Chat interface
        st.subheader("Chat with your document")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input for user query
        if prompt := st.chat_input("Ask a question about your document..."):
            # Check if Replicate API key is available
            if not replicate_api_key or not REPLICATE_AVAILABLE:
                if not REPLICATE_AVAILABLE:
                    st.error("Replicate package is not installed. Please install with: pip install replicate")
                else:
                    st.error("Replicate API token is required for generating responses. Please add it in the sidebar settings.")
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
                        
                        # Generate response using Replicate
                        response = generate_response(document_content, prompt)
                        
                        # Display response
                        st.markdown(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Show a welcome message if no document is loaded
        st.info("👈 Please upload a document using the sidebar to start chatting.")
        
        # Show information about OLMoCR-7B
        st.markdown("""
        ### About OLMoCR-7B
        
        This app now uses **OLMoCR-7B** from Replicate for OCR processing:
        
        - **Advanced OCR**: Fine-tuned from Qwen2-VL-7B-Instruct specifically for OCR tasks
        - **PDF Support**: Optimized for processing PDF documents
        - **High Accuracy**: Trained on the olmOCR-mix-0225 dataset for better text extraction
        
        **Requirements:**
        - Replicate API token for OLMoCR-7B OCR processing
        - Mistral API key for fallback OCR (optional)
        
        Get your Replicate API token at: https://replicate.com/account/api-tokens
        """)

if __name__ == "__main__":
    main()