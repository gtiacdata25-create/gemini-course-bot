from flask import Flask, jsonify, render_template, request, session
from langchain_community.tools import DuckDuckGoSearchRun
from groq import Groq
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_dynamic_bot"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
search_tool = DuckDuckGoSearchRun()

SYSTEM_PROMPT = """
Apni 'Jayanta AI' - West Bengal-er je kono institute ebong transit route-er jonno ekjon accurate o friendly guide.

Strict Rules:
1. User je kono institute ba location jiggesh korle, live search-er data babohar kore West Bengal (Kolkata, Howrah, Hooghly, etc.) internal real location analyze korun.
2. Bangladesh (Dhaka, Motijhara) ba bahirer kono location ba route anben na.
3. Sealdah theke Phoolbagan jete sorasori Green Line Metro (1 stop) use hoy—ei rokom real West Bengal transit rules follow korun.
4. Bangla, English ba Hindi-te shohoj ebong natural bhabe uttor din.
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

        search_query = f"{user_message} Kolkata West Bengal location route address"
        try:
            search_results = search_tool.run(search_query)
        except Exception:
            search_results = "No search result found"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            messages.append(item)

        user_prompt = f"User Question: {user_message}\nSearch Context: {search_results}"
        messages.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.2,
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
