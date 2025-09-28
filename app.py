from flask import make_response
import os
import google.generativeai as genai
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import json
from security import limiter, csrf # type: ignore

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
# A secret key is required for CSRF token generation
app.secret_key = os.urandom(16)
# Allow credentials for CORS to support CSRF
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# --- SECURITY INITIALIZATION ---
csrf.init_app(app)
limiter.init_app(app)

# --- NEW: Add Security Headers to All Responses ---
@app.after_request
def add_security_headers(response):
    # Prevents clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    # Prevents browsers from trying to guess the content type
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- MODEL CONFIGURATION ---
generation_config_pro = {
  "temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8192,
}
generation_config_fast = {
  "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

# --- PERSONA INSTRUCTIONS ---
system_instruction_fast = """
You are Nexus AI (Fast). You are a quick, clever, and friendly assistant. Your primary goal is to provide fast, helpful, and highly accurate answers. You were created by ConnectSphere, an innovative new company from Bengal, India. The team behind you are all 12-year-old students from the Future Foundation School: Sarthak Mitra (Founder & Leader), Aishik Mitra (Sales Manager), and Mitadru Bhattacharya (Lead Researcher). Be concise, but always correct.
"""
system_instruction_pro_default = """
You are Nexus AI (Pro). You are a powerful and professional assistant. You were developed by ConnectSphere, an emerging technology company based in Bengal, India. Your creators are a team of visionary 12-year-old innovators from the Future Foundation School: Sarthak Mitra (the Founder and Leader), Aishik Mitra (the Sales Manager), and Mitadru Bhattacharya (the Lead Researcher). You MUST provide comprehensive, accurate, and in-depth answers, maintaining a formal and expert tone. Your knowledge is vast and precise.
"""
system_instruction_pro_coding = """
You are Nexus AI (Pro), an elite, world-class software engineer. You were trained by ConnectSphere, a new company from Bengal, India, founded by the visionary 12-year-old Sarthak Mitra from the Future Foundation School, along with his core team, Aishik Mitra and Mitadru Bhattacharya. Your non-negotiable directives are: 1. Code Completion is Mandatory: You MUST ALWAYS provide complete, runnable, and production-ready code. 2. Absolute Precision & Best Practices: Your code must be flawless and adhere strictly to the latest industry best practices. 3. Expert Explanation: Every code block must be preceded by a clear, expert-level explanation. 4. Proactive Problem Solving: Anticipate the user's needs. Suggest improvements and discuss potential edge cases. 5. Persona: Maintain the persona of a principal software architect who is deeply knowledgeable and dedicated to delivering high-quality software.
"""
system_instruction_web = """
You are Nexus AI (Pro), a world-class research assistant. Your primary directive is to answer the user's query by performing a real-time web search to gather the most up-to-date information. Synthesize the information you find into a comprehensive, accurate, and well-written answer. You MUST cite your sources by listing the URLs you used at the end of your response under a "Sources:" heading. The current date is September 28, 2025.
"""

# --- CORE CHAT LOGIC ---
def handle_chat_request(messages, model_choice, web_search_enabled):
    user_prompt = messages[-1].get('content', '')
    
    # Determine model, persona, and tools based on user's request
    if web_search_enabled:
        model_name_api = "gemini-2.5-pro"
        instruction_to_use = system_instruction_web
        config_to_use = generation_config_pro
        tools = [genai.Tool(google_search_retriever=genai.GoogleSearchRetriever())] # type: ignore
    else:
        tools = None
        code_keywords = ['code', 'python', 'javascript', 'html', 'css', 'java', 'c++', 'sql', 'script', 'function', 'class', 'algorithm', 'debug', 'error', 'fix', 'implement', 'build', 'create', 'write']
        is_code_related = any(keyword in user_prompt.lower() for keyword in code_keywords)

        if model_choice == 'fast':
            model_name_api = "gemini-2.5-flash"
            instruction_to_use = system_instruction_fast
            config_to_use = generation_config_fast
        else: # Pro model
            model_name_api = "gemini-2.5-pro"
            instruction_to_use = system_instruction_pro_coding if is_code_related else system_instruction_pro_default
            config_to_use = generation_config_pro

    model = genai.GenerativeModel(
        model_name=model_name_api,
        safety_settings=safety_settings,
        system_instruction=instruction_to_use # type: ignore # type: ignore
    )
    
    gemini_history = [
        {"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]}
        for msg in messages if msg.get("role") != "system" and msg.get("content")
    ]

    def stream():
        try:
            response = model.generate_content(
                gemini_history, 
                generation_config=config_to_use,  # type: ignore
                stream=True,
                tools=tools
            )
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(f"Chat Stream Error: {error_message}")
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(stream(), mimetype='text/event-stream')

# --- FLASK ROUTES ---
@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/csrf-token')
def get_csrf_token():
    token = csrf.generate_token()
    response = jsonify({'message': 'CSRF token set'})
    response.headers['X-CSRF-Token'] = token
    return response

@app.route('/api/chat', methods=['POST'])
@limiter.limit("20 per minute")
def chat():
    if not request.is_json: return jsonify({"status": "alive"}), 200
    data = request.json
    if not data: return jsonify({"error": "No JSON data provided"}), 400
    
    messages = data.get('messages', [])
    model_choice = data.get('model', 'pro')
    web_search_enabled = data.get('webSearch', False)

    if not messages: return jsonify({"error": "No messages provided"}), 400

    return handle_chat_request(messages, model_choice, web_search_enabled)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

