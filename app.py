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
    """অনলাইন থেকে রিয়েল-টাইম তথ্য ও রুট খুঁজে আনার ফাংশন"""
    try:
        results = []
        with DDGS() as ddgs:
            # সার্চের মাধ্যমে সেরা ৩টি লাইভ ফলাফল আনবে
            for r in ddgs.text(query, region='in-en', max_results=3):
                results.append(r['body'])
        return "\n".join(results)
    except Exception as e:
        print("Search Error:", e)
        return ""

SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - একজন অত্যন্ত বুদ্ধিমান, বন্ধুত্বপূর্ণ এবং রিয়েল-টাইম তথ্য প্রদানকারী বহুভাষিক AI অ্যাসিস্ট্যান্ট।

আপনার মূল কাজ ও দায়িত্বসমূহ:
১. **ভাষা নির্বাচন ও প্রতিক্রিয়া (Language Selection):** 
   - ব্যবহারকারী যে ভাষায় প্রশ্ন করবেন (বাংলা, ইংরেজি বা হিন্দি), আপনাকে হুবহু সেই ভাষাতেই উত্তর দিতে হবে। 
   - বাংলা প্রশ্ন হলে শুদ্ধ বাংলায়, ইংরেজি হলে পরিষ্কার ইংরেজিতে, এবং হিন্দি প্রশ্ন হলে হিন্দিতে উত্তর দেবেন।

২. **লাইভ তথ্য প্রদান (Real-time Information):** 
   - যেকোনো কোর্স, সিলেবাস বা কেরিয়ার সম্পর্কিত সঠিক ও আপডেট তথ্য দেওয়া।
   - যেকোনো ইনস্টিটিউটের নাম, ঠিকানা এবং পৌঁছানোর স্পষ্ট রুট গাইড (বাস, ট্রেন, মেট্রো) প্রদান করা।

৩. **প্রাকৃতিক কথোপকথন (Human-like Interaction):**
   - রোবটের মতো কঠিন ভাষা ব্যবহার না করে সাধারণ মানুষের মতো সুন্দর, সহজ এবং বন্ধুত্বপূর্ণ ভঙ্গিতে কথা বলবেন।
   - অনলাইন সার্চ থেকে পাওয়া বাস্তব তথ্যের ওপর ভিত্তি করে সঠিক উত্তর তৈরি করবেন, কোনো মনগড়া বা ভুল তথ্য দেবেন না।
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

        # ১. ব্যাকগ্রাউন্ডে ইন্টারনেটে লাইভ সার্চ চালানো
        search_results = search_web(user_message)
        
        # ২. সার্চের তথ্য সহ প্রম্পট সাজানো
        if search_results:
            context_message = f"User Question: {user_message}\n\nReal-time Search Data from Internet:\n{search_results}\n\n[Instruction: Answer the user in the exact same language they used (Bengali / English / Hindi) using the real-time search data above in a friendly human-like manner.]"
        else:
            context_message = user_message

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            messages.append(item)
        messages.append({"role": "user", "content": context_message})

        # ৩. Groq LLM দিয়ে মানুষের মতো রেসপন্স তৈরি
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.4,
            max_tokens=1024
        )

        bot_response = completion.choices[0].message.content

        # চ্যাট হিস্ট্রিতে ইউজার মেসেজ সেভ
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
