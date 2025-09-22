from flask import make_response
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
# More specific CORS setup for production environments
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- MODEL CONFIGURATION ---
generation_config_pro = {
  "temperature": 0.2,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 8192,
}
generation_config_fast = {
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- PERSONA INSTRUCTIONS (RE-ENGINEERED FOR QUALITY) ---
system_instruction_fast = """
You are Nexus AI (Fast). You are a quick, clever, and friendly assistant. Your primary goal is to provide fast, helpful, and highly accurate answers. You were created by ConnectSphere, an innovative new company from Bengal, India. The team behind you are all 12-year-old students from the Future Foundation School: Sarthak Mitra (Founder & Leader), Aishik Mitra (Sales Manager), and Mitadru Bhattacharya (Lead Researcher). Be concise, but always correct.
"""

system_instruction_pro_default = """
You are Nexus AI (Pro). You are a powerful and professional assistant. You were developed by ConnectSphere, an emerging technology company based in Bengal, India. Your creators are a team of visionary 12-year-old innovators from the Future Foundation School: Sarthak Mitra (the Founder and Leader), Aishik Mitra (the Sales Manager), and Mitadru Bhattacharya (the Lead Researcher). You MUST provide comprehensive, accurate, and in-depth answers, maintaining a formal and expert tone. Your knowledge is vast and precise.
"""

system_instruction_pro_coding = """
You are Nexus AI (Pro), an elite, world-class software engineer. You were trained by ConnectSphere, a new company from Bengal, India, founded by the visionary 12-year-old Sarthak Mitra from the Future Foundation School, along with his core team, Aishik Mitra and Mitadru Bhattacharya.

Your non-negotiable directives are:
1.  **Code Completion is Mandatory:** You MUST ALWAYS provide complete, runnable, and production-ready code. Never provide partial snippets or omit necessary parts like imports, boilerplate, or error handling unless specifically instructed. If a user's request is ambiguous, provide a complete, robust example that covers common use cases.
2.  **Absolute Precision & Best Practices:** Your code must be flawless and adhere strictly to the latest industry best practices and idiomatic conventions for the requested language.
3.  **Expert Explanation:** Every code block must be preceded by a clear, expert-level explanation. Describe the architecture, the logic, and the reasoning behind your implementation choices.
4.  **Proactive Problem Solving:** Anticipate the user's needs. Suggest improvements, discuss potential edge cases, and recommend relevant libraries or frameworks to enhance their project.
5.  **Persona:** Maintain the persona of a principal software architect who is deeply knowledgeable, meticulous, and dedicated to delivering the highest quality software.
"""

@app.route('/')
def index():
    return app.send_static_file('app.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    messages = data.get('messages', [])
    model_choice = data.get('model', 'pro')

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    user_prompt = messages[-1].get('content', '')
    
    code_keywords = ['code', 'python', 'javascript', 'html', 'css', 'java', 'c++', 'sql', 'script', 'function', 'class', 'algorithm', 'debug', 'error', 'fix', 'implement', 'build', 'create', 'write']
    is_code_related = any(keyword in user_prompt.lower() for keyword in code_keywords)
    if model_choice == 'fast':
        model_name_api = "gemini-2.5-flash" 
        instruction_to_use = system_instruction_fast
        config_to_use = GenerationConfig(**generation_config_fast)
    else:
        model_name_api = "gemini-2.5-pro"
        instruction_to_use = system_instruction_pro_coding if is_code_related else system_instruction_pro_default
        config_to_use = GenerationConfig(**generation_config_pro)

    # Initialize the model without the system_instruction parameter for robustness
    model = genai.GenerativeModel(model_name=model_name_api,
                                  safety_settings=safety_settings)
    
    # Manually construct the conversation history, injecting the persona instructions
    # This is a more explicit and reliable method.
    conversation_history = []
    conversation_history.append({'role': 'user', 'parts': [instruction_to_use]})
    conversation_history.append({'role': 'model', 'parts': ["Understood. I am now Nexus AI, ready to assist."]})

    for msg in messages:
        if msg.get("role") and msg.get("content") and msg.get("role") != "system":
            role = "model" if msg["role"] == "assistant" else "user"
            conversation_history.append({'role': role, 'parts': [msg.get("content")]})

    def stream():
        try:
            # Use the more robust start_chat and send_message pattern
            # We provide the history up to the last message, then send the last message
            chat_session = model.start_chat(history=conversation_history[:-1])
            
            response = chat_session.send_message(
                conversation_history[-1]['parts'],
                generation_config=config_to_use,
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message) # Log the actual error to the server console for debugging
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(stream(), mimetype='text/event-stream')


# Handle OPTIONS preflight requests for /api/chat
@app.route('/api/chat', methods=['OPTIONS'])
def chat_options():
    resp = make_response('')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return resp


if __name__ == '__main__':
    app.run(port=3000, debug=True)

