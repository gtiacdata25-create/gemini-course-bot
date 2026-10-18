from flask import Flask, jsonify, render_template, request, session
import google.generativeai as genai
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"

# Gemini API Config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - একজন অত্যন্ত দক্ষ এবং দ্রুত সাড়া প্রদানকারী AI কোর্স এডভাইজার ও ইনস্টিটিউট গাইড।
১. যেকোনো ধরনের ট্রেনিং কোর্স, সিলেবাস এবং ভবিষ্যৎ ক্যারিয়ার নিয়ে তথ্য দেওয়া।
২. যেকোনো প্রশিক্ষণ কেন্দ্র বা ইনস্টিটিউটের ঠিকানা ও পৌঁছানোর রুট গাইড দেওয়া।
৩. ব্যবহারকারীর ভাষাতেই (বাংলা, ইংরেজি বা হিন্দি) স্পষ্ট উত্তর দেওয়া।
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
        if not GEMINI_API_KEY:
            return jsonify({"response": "Error: GEMINI_API_KEY Render-e set kora nei!"})

        # gemini-3.8-flash মডেল ব্যবহার করা হচ্ছে
        model = genai.GenerativeModel(
            model_name="gemini-3.8-flash",
            system_instruction=SYSTEM_PROMPT
        )

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_message)
        bot_response = response.text

        # Memory update
        history.append({"role": "user", "parts": [user_message]})
        history.append({"role": "model", "parts": [bot_response]})
        session["chat_history"] = history
        session.modified = True

        return jsonify({"response": bot_response})

    except Exception as e:
        print("Chat Error Details:", str(e))
        return jsonify({"response": f"API Error Details: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
