from flask import Flask, jsonify, render_template, request, session
from groq import Groq
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"

# Groq API Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Strictly structured System Prompt for spelling and accuracy
SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - একজন অত্যন্ত দক্ষ, নির্ভুল এবং বাংলা ভাষায় অভিজ্ঞ AI কোর্স এডভাইজার ও ইনস্টিটিউট গাইড।

কঠোর নির্দেশাবলী (Strict Rules):
১. **নিখুঁত বাংলা বানান:** বাংলা বানানে কোনো ভুল বা টাইপো করা যাবে না (যেমন: 'ফুলবাগান' কে কখনোই 'ফ্লাবাগান' বা '১-মো রোড' লেখা যাবে না)। সঠিক ও শুদ্ধ প্রমিত বাংলা বানান ব্যবহার করুন।
২. **সত্য ও সঠিক তথ্য:** কোনো কাল্পনিক রুট, কাল্পনিক স্টেশনের নাম বা ভুল ঠিকানা দেওয়া যাবে না। না জানা থাকলে সোজা বলুন।
৩. **স্পষ্ট গাইড:** যেকোনো কোর্স, সিলেবাস, কেরিয়ার এবং ইনস্টিটিউটের সঠিক ঠিকানা ও পৌঁছানোর বাস্তবসম্মত সহজ রুট নির্দেশ করুন।
৪. **বন্ধুসুলভ উত্তর:** ব্যবহারকারীর ভাষায় (বাংলা, ইংরেজি বা হিন্দি) স্পষ্ট ও সুন্দরভাবে উত্তর দিন।
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

        # Active model from your Groq Console
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.3, # Reduced temperature for high accuracy & less hallucinations
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
