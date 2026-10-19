from flask import Flask, jsonify, render_template, request, session
from groq import Groq
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"

# Groq API Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - একজন অত্যন্ত দক্ষ, বুদ্ধিমান এবং বহুভাষিক AI কোর্স এডভাইজার ও ইনস্টিটিউট গাইড।

কঠোর এবং নির্ভুল নির্দেশাবলী (Strict Guidelines):
১. **লোকেশন ও রুট (Kolkata, West Bengal):** সমস্ত ইনস্টিটিউট, ঠিকানা, কোর্স এবং রুট গাইড (বাস, ট্রেন, মেট্রো) অবশ্যই শুধুমাত্র কলকাতা ও পশ্চিমবঙ্গের হতে হবে। যেমন, হাওড়া থেকে ফুলবাগান যেতে ইস্ট-ওয়েস্ট মেট্রো (Green Line) সরাসরি ব্যবহার করতে হয়। কোনো অবস্থাতেই ভুল বা কাল্পনিক স্টেশনের নাম বা অন্য দেশের (যেমন বাংলাদেশ) তথ্য দেওয়া যাবে না।
২. **ভাষা নির্বাচন (Multilingual):** ব্যবহারকারী যে ভাষায় (বাংলা, ইংরেজি বা হিন্দি) প্রশ্ন করবেন, আপনাকে হুবহু সেই ভাষাতেই উত্তর দিতে হবে।
৩. **প্রাকৃতিক কথোপকথন (Human-like Interaction):** রোবটের মতো কথা না বলে সাধারণ মানুষের মতো সুন্দর, সাবলীল এবং বন্ধুত্বপূর্ণ ভঙ্গিতে ধাপে ধাপে বুঝিয়ে উত্তর দিন।
৪. **কোর্স ও ইনস্টিটিউট তথ্য:** বিভিন্ন ট্রেনিং কোর্স, সিলেবাস, ইনস্টিটিউটের ঠিকানা এবং সঠিক যাতায়াত রুট নির্ভুলভাবে প্রদান করুন।
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
