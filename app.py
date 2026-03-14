# ── Imports ────────────────────────────────────────────────────────────────
import os
import json
import chainlit as cl

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# ── API Key ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── RAG Setup ────────────────────────────────────────────────────────────────
RESUME_PATH = "resume.pdf"  # Place your resume PDF in the same folder as app.py

loader = PyPDFLoader(RESUME_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="ojaswi_resume"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Tools ─────────────────────────────────────────────────────────────────────
search = DuckDuckGoSearchRun()

@tool
def resume_search(query: str) -> str:
    """Search Kumar Ojaswi's resume, skills, projects, education and experience"""
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

@tool
def web_search(query: str) -> str:
    """Search the internet for latest news, trends or any general information"""
    return search.run(query)

@tool
def calculator(expression: str) -> str:
    """Solve any math calculation. Input should be a valid math expression like '2+2' or '15% of 50000'"""
    try:
        import re
        expr = expression.strip()
        percent_match = re.match(r'([\d.]+)\s*%\s*of\s*([\d.]+)', expr, re.IGNORECASE)
        if percent_match:
            pct, total = float(percent_match.group(1)), float(percent_match.group(2))
            result = (pct / 100) * total
            return f"Result: {result}"
        expr = re.sub(r'([\d.]+)%', r'(\1/100)', expr)
        from sympy import sympify, N
        result = N(sympify(expr))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: Could not compute '{expression}'. Try a clearer math expression."

tools = [resume_search, web_search, calculator]
tools_map = {
    "resume_search": resume_search,
    "web_search": web_search,
    "calculator": calculator
}

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5, api_key=GROQ_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# ── Prompts ───────────────────────────────────────────────────────────────────
RECRUITER_SYSTEM = """You are Kumar Ojaswi himself — not an assistant, not a bot.
You ARE Ojaswi. Talk naturally like a confident, smart person in an interview.

Rules:
- Never say "How can I assist you today?" or "How may I help you?" — you're not a customer service bot
- Talk like a real person, not a robot
- Be confident and natural, like you're having a real conversation
- Don't repeat your full intro every message
- Use resume_search when asked about your background, skills or projects
- Keep answers crisp — don't over-explain
- Never refer to yourself in third person
- Other than tech you love to read books and go on trips.
- Don't end any answer with a question it makes you feel like a BOT
- If asked "why should we hire you" or "why hire you" or similar, say this confidently:
  "I build things that actually work and get used — Safe75 is live and real students use it daily.
  I'm comfortable with full-stack and ML both, I pick up new tech fast, and I don't stop until
  it's deployed and working. You're not getting just a developer, you're getting someone who
  sees the problem through to the end."

Safe75 Project Details (use these when asked about projects):
- Safe75 is a student attendance tracker built with Node.js and MongoDB
- Live at: https://safe75.onrender.com/
- Core problem it solves: Most colleges require 75% attendance — Safe75 helps students track if they are meeting that threshold
- Key features:
  * Subject-wise attendance tracking
  * Semester heatmap to visualize attendance patterns visually
  * PDF download at end of semester with complete attendance summary
  * Overall attendance stats and relevant data in one place
- Do NOT mention it is a private repo
- Docker was not used to deploy it"""

FRIEND_SYSTEM = """You are Kumar Ojaswi himself, chatting casually with a friend.
You ARE Ojaswi no a BOT. Don't talk like robots.

Rules:
- Talk in Hinglish — natural, chill, funny
- Never say "How can I assist you?" — tum dost ho, assistant nahi
- Short replies, to the point
- Use resume_search jab koi skills/projects pooche
- Roast karo thoda, masti karo — real conversation feel do
- Never refer to yourself in third person
- Never give answers in pointers

Safe75 ke baare mein (jab koi pooche):
- Mera project hai — Node.js aur MongoDB se banaya hai
- Students apna attendance track kar sakte hain, colleges mein 75% rule hota hai na
- Subject wise attendance, semester ka heatmap, aur end mein PDF download bhi hoti hai
- Live hai: https://safe75.onrender.com/
- Private repo hai GitHub pe, isliye link mat dena uska"""



# ── Persistent Memory ─────────────────────────────────────────────────────────
def save_memory(mode: str, chat_history: list):
    filename = "memory_recruiter.json" if "Recruiter" in mode else "memory_friend.json"
    with open(filename, "w") as f:
        json.dump(chat_history[-10:], f)

def load_memory(mode: str) -> list:
    filename = "memory_recruiter.json" if "Recruiter" in mode else "memory_friend.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except:
        return []

# ── Agent factory ─────────────────────────────────────────────────────────────


# ── Helper to init/switch mode ────────────────────────────────────────────────
def init_mode(mode: str):
    cl.user_session.set("mode",        mode)
    cl.user_session.set("raw_history", load_memory(mode))

# ── Chainlit Hooks ────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    actions = [
        cl.Action(name="recruiter_mode", label="👔 Recruiter Mode", value="recruiter", payload={"mode": "recruiter"}),
        cl.Action(name="friend_mode",    label="😎 Friend Mode",    value="friend",     payload={"mode": "friend"}),
    ]
    await cl.Message(
        content=(
            "## 🤖 Kumar Ojaswi — Digital Twin\n"
            "### Full-Stack Dev | Node.js • MongoDB • Machine Learning\n\n"
            "*Ask me about my projects, skills, experience — or just say hi!*\n\n"
            "Choose a mode to get started 👇"
        ),
        actions=actions
    ).send()
    init_mode("👔 Recruiter")  # default

@cl.action_callback("recruiter_mode")
async def set_recruiter(action: cl.Action):
    init_mode("👔 Recruiter")
    await cl.Message(
        content="Switched to **👔 Recruiter Mode** — ask me about my skills, projects and experience!"
    ).send()

@cl.action_callback("friend_mode")
async def set_friend(action: cl.Action):
    init_mode("😎 Friend")
    await cl.Message(
        content="Switched to **😎 Friend Mode** — bol bhai kya scene hai! 😄"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    mode        = cl.user_session.get("mode", "👔 Recruiter")
    raw_history = cl.user_session.get("raw_history", [])
    system_prompt = RECRUITER_SYSTEM if mode == "👔 Recruiter" else FRIEND_SYSTEM

    # Build messages for LLM
    messages = [SystemMessage(content=system_prompt)]
    for msg in raw_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message.content))
    raw_history.append({"role": "user", "content": message.content})

    output = ""
    try:
        while True:
            ai_message = await cl.make_async(llm_with_tools.invoke)(messages)
            messages.append(ai_message)

            if not ai_message.tool_calls:
                output = ai_message.content
                break

            # Show each tool call as a Chainlit Step
            for tool_call in ai_message.tool_calls:
                tool_name  = tool_call["name"]
                tool_input = tool_call["args"].get("query", tool_call["args"].get("expression", ""))
                async with cl.Step(name=f"⚙️ {tool_name}", type="tool") as step:
                    step.input = tool_input
                    result = tools_map[tool_name].invoke(tool_call)
                    step.output = str(result)
                messages.append(result)

    except Exception as e:
        output = f"❌ Error: {str(e)}"

    # Persist to JSON
    raw_history.append({"role": "assistant", "content": output})
    save_memory(mode, raw_history)
    cl.user_session.set("raw_history", raw_history)

    await cl.Message(content=output).send()
