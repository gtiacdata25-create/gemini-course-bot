from flask import Flask, jsonify, render_template, request, session
from groq import Groq
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"

# Groq API Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - একজন অত্যন্ত দক্ষ এবং দ্রুত সাড়া প্রদানকারী AI কোর্স এডভাইজার ও ইনস্টিটিউট গাইড।
১. যেকোনো ধরনের ট্রেনিং কোর্স, সিলেবাস এবং ভবিষ্যৎ ক্যারিয়ার নিয়ে তথ্য দেওয়া।
২. যেকোনো প্রশিক্ষণ কেন্দ্র বা ইনস্টিটিউটের ঠিকানা ও পৌঁছানোর রুট গাইড দেওয়া।
৩. ব্যবহারকারীর ভাষাতেই (বাংলা, ইংরেজি বা হিন্দি) স্পষ্ট ও বন্ধুসুলভ উত্তর দেওয়া।
"""

@app.route("/")
def home():
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "অনুগ্রহ করে একটি প্রশ্ন লিখুন।"})

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

        # Groq Llama 3.3 70B Versatile
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
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
        # Asli error-ti dekhano hocche
        return jsonify({"response": f"API Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
