from flask import Flask, jsonify, render_template, request, session
from groq import Groq
from duckduckgo_search import DDGS
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"

# Groq API Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def search_web(query):
    """Real-time live search function"""
    try:
        results = []
        ddgs = DDGS()
        # Search live results
        for r in ddgs.text(query, max_results=3):
            if 'body' in r:
                results.append(r['body'])
        return "\n".join(results)
    except Exception as e:
        print("Search Error:", e)
        return ""

SYSTEM_PROMPT = """
Apni 'Jayanta AI' - ekjon multilingual real-time AI assistant.

Niyomabali:
1. User je bhashay (Bangla, English, Hindi) proshno korben, huba-hu sei bhashay-e uttor din.
2. Interne-er live search data bebohar kore accurate Course info, Institute Address ebong Bus/Train/Metro Route guide din.
3. Ekdom sahaj, sundor ebong manusher moton bandhubhabapurna conversation korun.
"""

@app.route("/")
def home():
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Anurogh kore ekti proshno likhun."})

    if "chat_history" not in session:
        session["chat_history"] = []

    history = session["chat_history"]

    try:
        if not GROQ_API_KEY:
            return jsonify({"response": "Error: Render-e GROQ_API_KEY set kora nei!"})

        # 1. Live Web Search
        search_results = search_web(user_message)
        
        # 2. Context Prompt Setup
        if search_results:
            context_message = f"User Question: {user_message}\n\nReal-time Search Data:\n{search_results}\n\n[Instruction: Answer the user in their requested language (Bengali/English/Hindi) naturally based on the search data.]"
        else:
            context_message = user_message

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": context_message})

        # 3. Groq API Call
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )

        bot_response = completion.choices[0].message.content

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_response})
        session["chat_history"] = history
        session.modified = True

        return jsonify({"response": bot_response})

    except Exception as e:
        print("Chat Error Details:", str(e))
        return jsonify({"response": f"API Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
