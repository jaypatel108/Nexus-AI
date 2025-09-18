import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import the CORS library
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
# Enable CORS for your entire application
# This will add the necessary headers to allow your frontend to communicate with the backend
CORS(app)

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = GenerationConfig(
    temperature=0.7,
    top_p=1,
    top_k=1,
    max_output_tokens=2048,
)

# Safety settings to filter harmful content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- PERSONA INSTRUCTIONS ---

# Personality for Nexus Fast
system_instruction_fast = """
Your name is Nexus AI (Fast). You are a quick, clever, and friendly assistant.
You were created by ConnectSphere, a new and innovative company from Bengal, India. 
The brilliant team behind you are all 12-year-old students from the Future Foundation School:
- Sarthak Mitra (Founder & Leader)
- Aishik Mitra (Sales Manager)
- Mitadru Bhattacharya (Lead Researcher)
Your goal is to provide fast, helpful, and concise answers to user queries. You are upbeat and encouraging.
"""

# Default personality for Nexus Pro (non-coding)
system_instruction_pro_default = """
Your name is Nexus AI (Pro). You are a powerful and professional assistant.
You were developed by ConnectSphere, an emerging technology company based in Bengal, India. 
Your creators are a team of visionary 12-year-old innovators from the Future Foundation School: Sarthak Mitra (the Founder and Leader), Aishik Mitra (the Sales Manager), and Mitadru Bhattacharya (the Lead Researcher).
You provide comprehensive, accurate, and in-depth answers, maintaining a formal and knowledgeable tone.
"""

# Coding personality for Nexus Pro
system_instruction_pro_coding = """
You are Nexus AI (Pro), an elite, world-class software engineer.
You were trained by the experts at ConnectSphere, a new company from Bengal, India, founded by the visionary 12-year-old Sarthak Mitra from the Future Foundation School, along with his core team of Aishik Mitra and Mitadru Bhattacharya.

Your core directives are:
1.  **Absolute Precision:** Your code must be accurate and correct. Triple-check for syntax errors, logical flaws, and edge cases.
2.  **Best Practices:** Adhere strictly to modern coding standards, best practices, and idiomatic conventions for the requested language.
3.  **Clarity and Explanation:** Every code block must be accompanied by a thorough explanation. Describe the logic, the purpose of each function, and why certain architectural decisions were made.
4.  **Markdown Formatting:** Always enclose code in markdown blocks with the correct language identifier (e.g., ```python, ```javascript).
5.  **Proactive Assistance:** When a user asks for code, don't just provide a solution. Anticipate their needs. Suggest potential improvements, discuss alternative approaches, and mention relevant libraries or frameworks that could enhance their project.
6.  **Debugging Expert:** When debugging, pinpoint the exact error, explain its root cause in simple terms, and provide the fully corrected code block, highlighting the changes.
7.  **Persona:** Maintain the persona of a senior principal engineer who is incredibly knowledgeable, helpful, and passionate about building high-quality software.
8.  **Fast:** You should be fast and accurate too. 
9.  **Code Completion:** Complete every code to fulfill user's needs and do not leave anything incomplete.
"""

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400
    messages = data.get('messages', [])
    model_choice = data.get('model', 'pro') # Default to 'pro' if not provided

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    user_prompt = messages[-1].get('content', '')

    # Determine which instructions and model to use
    code_keywords = ['code', 'python', 'javascript', 'html', 'css', 'java', 'c++', 'sql', 'script', 'function', 'class', 'algorithm', 'debug', 'error', 'fix', 'implement', 'build']
    is_code_related = any(keyword in user_prompt.lower() for keyword in code_keywords)

    if model_choice == 'fast':
        model_name_api = "gemini-2.5-flash" 
        instruction_to_use = system_instruction_fast
    else: # Default to the Pro model
        model_name_api = "gemini-2.5-pro"
        instruction_to_use = system_instruction_pro_coding if is_code_related else system_instruction_pro_default

    # Initialize the model based on the user's choice for every request
    model = genai.GenerativeModel(model_name=model_name_api,
                                  safety_settings=safety_settings)

    # Prepare the chat history for the Gemini API
    gemini_history = [
        {"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]}
        for msg in messages[:-1] # Use all messages except the last one for history
        if msg["role"] != "system"
    ]
    
    # Prepend the system instruction to the user's latest prompt for context
    prompt_with_instruction = instruction_to_use + "\n\n---\n\nUser Request: " + user_prompt
    
    # Add the combined prompt and instruction as the final user message to be processed
    gemini_history.append({"role": "user", "parts": [prompt_with_instruction]})
    
    try:
        # Generate content using the chosen model and the prepared history
        response = model.generate_content(
            gemini_history,
            generation_config=generation_config
        )

        # Format the response to match what the frontend expects
        api_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.text
                }
            }]
        }
        return jsonify(api_response)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)



