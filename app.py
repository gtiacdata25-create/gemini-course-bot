import os
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# API Key ক্লাউড এনভায়রনমেন্ট (Render) থেকে নেওয়া হবে
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")
    language = data.get("language", "Bengali")
    chat_history = data.get("history", [])
    
    system_instruction = (
        f"You are an expert Training Course Advisor. Provide 100% accurate, precise, "
        f"and structured information about training courses. "
        f"You must reply strictly in the following language: {language}."
    )

    formatted_contents = []
    for msg in chat_history:
        role = "user" if msg.get("sender") == "user" else "model"
        formatted_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg.get("text"))]
            )
        )
    
    formatted_contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)