import streamlit as st
import base64
import tempfile
import os
import requests
import json
from PIL import Image
import io
import google.generativeai as genai
import time

# Load environment variables for deployment
def load_config():
    """Load configuration from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        app_id = st.secrets.get('MATHPIX_APP_ID', '')
        app_key = st.secrets.get('MATHPIX_APP_KEY', '')
        google_api_key = st.secrets.get('GOOGLE_API_KEY', '')
    except:
        # Fallback to environment variables (for other platforms)
        app_id = os.getenv('MATHPIX_APP_ID', '')
        app_key = os.getenv('MATHPIX_APP_KEY', '')
        google_api_key = os.getenv('GOOGLE_API_KEY', '')
    
    return app_id, app_key, google_api_key

# Global API keys
app_id, app_key, google_api_key = load_config()

# Test Mathpix API
def test_mathpix_api(app_id, app_key):
    """Test Mathpix API credentials"""
    if not app_id or not app_key:
        return False, "Missing credentials"
    
    headers = {
        "app_id": app_id,
        "app_key": app_key
    }
    
    try:
        # Test with a simple request to check credentials
        response = requests.get(
            "https://api.mathpix.com/v3/pdf-types",
            headers=headers
        )
        if response.status_code == 200:
            return True, "Connected successfully"
        else:
            return False, f"Authentication failed: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

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

# OCR Processing Functions with Mathpix
def process_image_with_mathpix(image_data, app_id, app_key):
    """Process image with Mathpix API"""
    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-type": "application/json"
    }
    
    # Prepare the request data
    data = {
        "src": f"data:image/png;base64,{image_data}",
        "formats": ["text", "md"],
        "data_options": {
            "include_asciimath": True,
            "include_latex": True
        }
    }
    
    try:
        response = requests.post(
            "https://api.mathpix.com/v3/text",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("md", result.get("text", ""))
        else:
            raise Exception(f"Mathpix API error: {response.status_code} - {response.text}")
    
    except Exception as e:
        raise Exception(f"Error processing image with Mathpix: {str(e)}")

def process_pdf_with_mathpix(pdf_content, filename, app_id, app_key):
    """Process PDF with Mathpix API"""
    headers = {
        "app_id": app_id,
        "app_key": app_key
    }
    
    # Step 1: Upload PDF
    files = {
        'file': (filename, pdf_content, 'application/pdf')
    }
    
    options = {
        'options_json': json.dumps({
            'conversion_formats': {
                'md': True,
                'docx': False
            },
            'math_inline_delimiters': ['$', '$'],
            'math_display_delimiters': ['$$', '$$']
        })
    }
    
    try:
        # Upload the PDF
        upload_response = requests.post(
            "https://api.mathpix.com/v3/pdf",
            headers=headers,
            files=files,
            data=options
        )
        
        if upload_response.status_code != 200:
            raise Exception(f"PDF upload failed: {upload_response.status_code} - {upload_response.text}")
        
        pdf_id = upload_response.json().get("pdf_id")
        
        # Step 2: Poll for completion
        max_attempts = 60  # Maximum 5 minutes
        attempt = 0
        
        while attempt < max_attempts:
            status_response = requests.get(
                f"https://api.mathpix.com/v3/pdf/{pdf_id}",
                headers=headers
            )
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "completed":
                    # Get the markdown content
                    md_response = requests.get(
                        f"https://api.mathpix.com/v3/pdf/{pdf_id}.md",
                        headers=headers
                    )
                    
                    if md_response.status_code == 200:
                        return md_response.text
                    else:
                        raise Exception(f"Failed to get markdown: {md_response.status_code}")
                
                elif status == "error":
                    error_info = status_data.get("error", "Unknown error")
                    raise Exception(f"PDF processing error: {error_info}")
                
                # Still processing, wait and retry
                time.sleep(5)
                attempt += 1
            else:
                raise Exception(f"Status check failed: {status_response.status_code}")
        
        raise Exception("PDF processing timeout - took too long")
    
    except Exception as e:
        raise Exception(f"Error processing PDF with Mathpix: {str(e)}")

def process_url_with_mathpix(url, app_id, app_key):
    """Process document from URL with Mathpix API"""
    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-type": "application/json"
    }
    
    data = {
        "url": url,
        "formats": ["text", "md"],
        "data_options": {
            "include_asciimath": True,
            "include_latex": True
        }
    }
    
    try:
        response = requests.post(
            "https://api.mathpix.com/v3/text",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("md", result.get("text", ""))
        else:
            raise Exception(f"Mathpix API error: {response.status_code} - {response.text}")
    
    except Exception as e:
        raise Exception(f"Error processing URL with Mathpix: {str(e)}")

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
    global app_id, app_key, google_api_key
    
    st.set_page_config(page_title="Document OCR & Chat", layout="wide")
    
    # Sidebar: Authentication for API keys
    with st.sidebar:
        st.header("Settings")
        
        # API key inputs - only show if not already configured
        if not app_id or not app_key or not google_api_key:
            st.warning("⚠️ API keys need to be configured for the app to work properly.")
            
            # API key inputs
            api_key_tab1, api_key_tab2 = st.tabs(["Mathpix API", "Google API"])
            
            with api_key_tab1:
                user_app_id = st.text_input("Mathpix App ID", value=app_id if app_id else "", type="password")
                if user_app_id and user_app_id != app_id:
                    app_id = user_app_id
                
                user_app_key = st.text_input("Mathpix App Key", value=app_key if app_key else "", type="password")
                if user_app_key and user_app_key != app_key:
                    app_key = user_app_key
            
            with api_key_tab2:
                user_google_api_key = st.text_input(
                    "Google API Key", 
                    value=google_api_key if google_api_key else "", 
                    type="password",
                    help="API key for Google Gemini to use for response generation"
                )
                if user_google_api_key and user_google_api_key != google_api_key:
                    google_api_key = user_google_api_key
        
        # Mathpix API validation
        mathpix_connected = False
        if app_id and app_key:
            is_valid, message = test_mathpix_api(app_id, app_key)
            if is_valid:
                st.sidebar.success(f"✅ Mathpix API {message}")
                mathpix_connected = True
            else:
                st.sidebar.error(f"❌ Mathpix API: {message}")
        
        # Google API key validation
        if google_api_key:
            is_valid, message = test_google_api(google_api_key)
            if is_valid:
                st.sidebar.success(f"✅ Google API {message}")
            else:
                st.sidebar.error(f"❌ Google API: {message}")
                google_api_key = None
        
        # Display warnings for missing API keys
        if not app_id or not app_key or not mathpix_connected:
            st.sidebar.warning("⚠️ Valid Mathpix API credentials required for document processing")
        
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
        
        # Only show document upload if Mathpix is connected
        if mathpix_connected:
            input_method = st.radio("Select Input Type:", ["PDF Upload", "Image Upload", "URL"])
            
            if input_method == "URL":
                url = st.text_input("Document URL:")
                if url and st.button("Load Document from URL"):
                    with st.spinner("Processing document from URL..."):
                        try:
                            content = process_url_with_mathpix(url, app_id, app_key)
                            
                            st.session_state.document_content = content
                            st.session_state.display_content = content
                            st.session_state.document_loaded = True
                            
                            st.success(f"Document processed successfully! Extracted {len(content)} characters.")
                        
                        except Exception as e:
                            st.error(f"Error processing URL: {str(e)}")
            
            elif input_method == "PDF Upload":
                uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
                if uploaded_file and st.button("Process PDF"):
                    content = uploaded_file.read()
                    
                    with st.spinner("Processing PDF... This may take a few moments."):
                        try:
                            # Process with Mathpix
                            extracted_content = process_pdf_with_mathpix(
                                content, 
                                uploaded_file.name,
                                app_id,
                                app_key
                            )
                            
                            st.session_state.document_content = extracted_content
                            st.session_state.display_content = extracted_content
                            st.session_state.document_loaded = True
                            
                            # Show success message and download option
                            st.success(f"✅ PDF processed successfully! Extracted {len(extracted_content)} characters.")
                            
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
                        
                        with st.spinner("Processing image..."):
                            # Process with Mathpix
                            extracted_content = process_image_with_mathpix(
                                img_str,
                                app_id,
                                app_key
                            )
                            
                            st.session_state.document_content = extracted_content
                            st.session_state.display_content = extracted_content
                            st.session_state.document_loaded = True
                            
                            st.success(f"Image processed successfully! Extracted {len(extracted_content)} characters.")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    # Main area: Display chat interface
    st.title("Document OCR & Chat")
    
    # Document preview area
    if "document_loaded" in st.session_state and st.session_state.document_loaded:
        with st.expander("Document Content", expanded=False):
            # Show the display version
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