from flask import Flask, jsonify, render_template, request, session
from langchain_community.tools import DuckDuckGoSearchRun
from groq import Groq
import os

app = Flask(__name__)
app.secret_key = "jayanta_ai_secret_key_dynamic_bot"

# Groq API Config
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
search_tool = DuckDuckGoSearchRun()

SYSTEM_PROMPT = """
আপনি 'Jayanta AI' - যেকোনো ইনস্টিটিউট ও লোকেশনের জন্য একটি রিয়েল-টাইম, গতিশীল এবং অতি-নির্ভুল গাইড।

কঠোর নীতিমালা (Strict Rules):
১. **যেকোনো অজানা ইনস্টিটিউটের অনুসন্ধান:** ব্যবহারকারী পশ্চিমবঙ্গের যেকোনো ইনস্টিটিউট, কলেজ বা স্কুলের নাম বললে, লাইভ ইন্টারনেটের তথ্য ব্যবহার করে তার সঠিক অবস্থান (যেমন: ফুলবাগান, কলকাতা) চিহ্নিত করুন।
২. **পশ্চিমবঙ্গের ভূগোল ভিত্তিক অবস্থান (Geo-Fencing):** যেকোনো অবস্থান বা ইনস্টিটিউট অবশ্যই ভারতের পশ্চিমবঙ্গের (Kolkata, Howrah, Hooghly ইত্যাদি) সীমানার ভেতরের হতে হবে। বাংলাদেশের কোনো অবস্থান বা নাম আনবেন না।
৩. **আসল ট্রানজিট রুট (Kolkata Transit Integration):**
   - **গ্রিন লাইন মেট্রো:** হাওড়া ময়দান ↔ হাওড়া ↔ শিয়ালদহ ↔ ফুলবাগান ↔ সল্টলেক সেক্টর V (হাওড়া বা শিয়ালদহ থেকে ফুলবাগান যেতে সরাসরি গ্রিন লাইন মেট্রো ব্যবহার করতে হয়)।
   - **ব্লু লাইন মেট্রো:** দক্ষিণেশ্বর ↔ দমদম ↔ এসপ্ল্যানেড ↔ কবি সুভাষ।
   - **ট্রেন রুট:** হাওড়া/শিয়ালদহ থেকে সংশ্লিষ্ট লোকাল ট্রেন ও অটো/টোটো কানেক্টিভিটি।
৪. **স্বাভাবিক ও বন্ধুভাবাপন্ন আচরণ:** ব্যবহারকারী সংক্ষেপে প্রশ্ন করলেও (যেমন: "শিয়ালদহ থেকে গ্লোবাল টেকনিক্যাল ইনস্টিটিউট কিভাবে যাব") নিজে থেকেই অবস্থান চিহ্নিত করে সাবলীল বাংলা, ইংরেজি বা হিন্দিতে ধাপ-ধাপে রুট বুঝিয়ে দিন।
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

        # Step 1: Dynamic Live Search for the user query
        search_query = f"{user_message} Kolkata West Bengal location route address"
        try:
            search_results = search_tool.run(search_query)
        except Exception:
            search_results = "No search result found"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            messages.append(item)

        # Context-rich prompt passing search results to LLM
        user_prompt = f"User Question: {user_message}\nSearch Information: {search_results}"
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
