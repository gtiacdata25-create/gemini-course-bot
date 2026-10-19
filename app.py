from flask import Flask, jsonify, render_template, request, session
from groq import Groq
import urllib.parse
import json
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_clean_bot"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Lightweight Live Search Tool without extra heavy libraries
def live_web_search(query):
    try:
        import requests
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.text[:2000] # Pass raw text context safely
    except Exception:
        pass
    return "No search data"

SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - যেকোনো ইনস্টিটিউট ও লোকেশনের জন্য একটি রিয়েল-টাইম, গতিশীল এবং অতি-নির্ভুল গাইড।

কঠোর নীতিমালা (Strict Rules):
১. **ইনস্টিটিউট অনুসন্ধান:** ব্যবহারকারী পশ্চিমবঙ্গের যেকোনো ইনস্টিটিউট (যেমন: Global Technical Institute, Nabagram Vidyapith ইত্যাদি)-এর নাম বললে লাইভ সার্চ কনটেক্সট ব্যবহার করে তার সঠিক অবস্থান (যেমন: ফুলবাগান, কলকাতা / কোননগর, হুগলি) চিহ্নিত করুন।
২. **জিও-ফেনসিং (Geo-Fencing):** ইনস্টিটিউটের সমস্ত তথ্য কেবল ভারতের পশ্চিমবঙ্গের (Kolkata, Howrah, Hooghly, etc.) সীমানার ভেতরে হতে হবে।
৩. **আসল ট্রানজিট রুট (Transit Rules):**
   - **গ্রিন লাইন মেট্রো:** হাওড়া ময়দান ↔ হাওড়া ↔ শিয়ালদহ ↔ ফুলবাগান ↔ সল্টলেক সেক্টর V (শিয়ালদহ বা হাওড়া থেকে ফুলবাগান যেতে সরাসরি গ্রিন লাইন মেট্রো)।
   - **ব্লু লাইন মেট্রো:** দক্ষিণেশ্বর ↔ দমদম ↔ এসপ্ল্যানেড ↔ কবি সুভাষ।
   - **ট্রেন রুট:** হাওড়া/শিয়ালদহ থেকে লোকাল ট্রেন ও অটো/টোটো কানেক্টিভিটি।
৪. **সহজ ও বন্ধুত্বপূর্ণ উত্তর:** সংক্ষেপে প্রশ্ন করলেও সম্পূর্ণ রুট ধাপে ধাপে সুন্দর বাংলা, ইংরেজি বা হিন্দিতে বুঝিয়ে দিন।
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

        # Dynamic search call
        search_query = f"{user_message} Kolkata West Bengal institute location address"
        search_context = live_web_search(search_query)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            messages.append(item)

        user_prompt = f"User Query: {user_message}\nSearch Snippet Context: {search_context}"
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
