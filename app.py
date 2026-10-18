from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_course_bot"  # চ্যাট মেমোরি ধরে রাখার জন্য সিক্রেট কী

# Gemini API Config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# System Prompt for Jayanta AI
SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - একজন অত্যন্ত দক্ষ এবং দ্রুত সাড়া প্রদানকারী AI কোর্স এডভাইজার ও ইনস্টিটিউট গাইড।

আপনার প্রধান কাজ:
১. যেকোনো ধরনের ট্রেনিং কোর্স, সিলেবাস এবং ভবিষ্যৎ ক্যারিয়ার নিয়ে শিক্ষার্থীকে দ্রুত ও নিখুঁত তথ্য দেওয়া।
২. যেকোনো প্রশিক্ষণ কেন্দ্র বা ইনস্টিটিউটের ঠিকানা, কিভাবে পৌঁছাতে হবে (বাস, ট্রেন, বা অন্য কোনো রুট গাইড বিবরণ) সুন্দর ও সহজ ভাষায় বুঝিয়ে দেওয়া।
৩. ব্যবহারকারী যে ভাষায় প্রশ্ন করবেন (বাংলা, ইংরেজি বা হিন্দি), আপনাকে ঠিক সে ভাষাতেই বন্ধুসুলভভাবে স্পষ্ট ও দ্রুত উত্তর দিতে হবে।
"""

@app.route("/")
def home():
    session.clear()  # নতুন সেশনে নতুন চ্যাট শুরু
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "অনুগ্রহ করে একটি প্রশ্ন লিখুন।"})

    # সেশন মেমোরি হ্যান্ডলিং (পূর্বের কথা মনে রাখার জন্য)
    if "chat_history" not in session:
        session["chat_history"] = []

    history = session["chat_history"]

    try:
        # দ্রুততম মডেল ব্যবহার করা হচ্ছে
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        
        # পূর্বে পাঠানো বার্তাগুলোর সমন্বয়ে চ্যাট সেশন চালু
        chat_session = model.start_chat(history=history)
        
        # দ্রুত রেসপন্স জেনারেট করা
        response = chat_session.send_message(user_message)
        bot_response = response.text

        # নতুন বার্তার হিস্ট্রি সেশনে সেভ করে রাখা
        history.append({"role": "user", "parts": [user_message]})
        history.append({"role": "model", "parts": [bot_response]})
        session["chat_history"] = history
        session.modified = True

        return jsonify({"response": bot_response})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": "দুঃখিত, সংযোগে সামান্য বিলম্ব হচ্ছে। অনুগ্রহ করে আবার চেষ্টা করুন।"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
