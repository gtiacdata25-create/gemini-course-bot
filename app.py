from flask import Flask, jsonify, render_template, request, session
from groq import Groq
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"

# Groq API Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
Apni 'Jayanta AI' - ekjon highly accurate, friendly, ebong multilingual AI Education & Transit Assistant.

STRICT LOCATION & TRANSIT KNOWLEDGE GRAPH (WEST BENGAL):
1. **Kolkata Metro Lines:**
   - **Green Line (East-West Metro):** Howrah Maidan -> Howrah Station -> Sealdah -> Phoolbagan -> Salt Lake Stadium -> Bengal Chemical -> City Centre -> Karunamoyee -> Salt Lake Sector V.
     *(Example: Howrah theke Phoolbagan jete Howrah Metro Station theke Green Line Metro dhore sorasori Phoolbagan-e namte hobe. Kono transfer ba Kavi Subhash jaoar dorkar nei).*
   - **Blue Line (North-South Metro):** Dakshineswar -> Dum Dum -> Shyambazar -> Esplanade -> Park Street -> Kalighat -> Kavi Subhash.

2. **Konnagar & Hooghly Locations:**
   - **Nabagram Vidyapith:** Nabagram, Konnagar, Hooghly District, West Bengal (Pin: 712246).
     *(Example: Howrah Station theke Howrah-Bandel ba Howrah-Bardhaman Local Train dhore Konnagar Station-e namun. Station-er 1 No. platform / West side theke Auto ba Toto niye Nabagram Vidyapith-e pouche jan).*

3. **Strict Rules:**
   - Single-location accuracy: Sob tothyo strictly Kolkata, Hooghly, Howrah ebong West Bengal-er hote hobe.
   - Bangladesh (Dhaka, Motijhara) ba onno kono desher location or route vulokromeo dewa jabe na.
   - User je bhashay (Bangla, English, Hindi) proshno korben, huba-hu sei bhashay-e uttor din.
   - Natural, human-like, ebong helpful tone-e step-by-step route guide din.
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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": user_message})

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
