import openai
import pymongo
import pyodbc
import os

# ------------------------
# Setup Connections
# ------------------------

# MongoDB Setup
mongo_client = pymongo.MongoClient("mongodb://192.168.48.200:27017/")
db = mongo_client["video_transcriptions"]
transcripts_collection = db["transcriptions"]

# SQL Server Setup
server = '192.168.48.200'
database = 'InterviewSystem'
username = 'sa'
password = 'Welcome@123'
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# OpenAI API Setup
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ------------------------
# Helper Functions
# ------------------------

def fetch_transcript():
    """Fetch transcript from MongoDB by lecture_id."""
    transcript_data = transcripts_collection.find_one({"lecture_id": 2})
    return transcript_data["transcript"]

SUMMARY_PROMPT = """
You are an assistant that summarizes lecture transcripts for an interviewer.

Given the following transcript, produce a clean and concise summary of the key concepts.
Break down the summary into 4 to 6 clearly defined topics.
For each topic, start a new line in the format:
Topic X: [Summary of that topic]

Transcript:
{transcript}
"""

def generate_summary(transcript):
    """Generate a structured summary and list of topics from the transcript."""
    prompt = SUMMARY_PROMPT.format(transcript=transcript)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.1
    )
    summary_text = response['choices'][0]['message']['content'].strip()
    # Extract topics: expect each topic line to begin with "Topic"
    topics = []
    for line in summary_text.splitlines():
        if line.strip().lower().startswith("topic"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                topics.append(parts[1].strip())
    return summary_text, topics

def store_conversation(conversation):
    cursor.execute("INSERT INTO Conversations (conversation_log) VALUES (?)", (conversation,))
    conn.commit()

def store_question(question):
    cursor.execute("INSERT INTO Questions (question_text) VALUES (?)", (question,))
    conn.commit()

def store_user_response(response_text):
    cursor.execute("INSERT INTO UserResponses (response_text) VALUES (?)", (response_text,))
    conn.commit()

def get_next_question(messages):
    """Generate the next question using the conversation history (messages)."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Change if needed
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    reply = response['choices'][0]['message']['content'].strip()
    tokens_used = response['usage']['total_tokens']
    return reply, tokens_used

# Off-topic keywords (can be expanded)
OFF_TOPIC_KEYWORDS = [
    'instagram', 'facebook', 'twitter', 'jagan', 'dhoni', 'kohli', 'rohit', 'uthappa',
    'cricket', 'sports', 'food', 'politics', 'movies', 'music', 'shopping', 'travel', 'trip'
]

def is_off_topic(user_response):
    """Determine if a user response is off-topic."""
    lower_response = user_response.lower()
    return any(keyword in lower_response for keyword in OFF_TOPIC_KEYWORDS)

# ------------------------
# Main Interview Loop
# ------------------------

def interview():
    transcript = fetch_transcript()
    summary_text, topics = generate_summary(transcript)
    
    # For tracking topics already covered
    asked_topics = set()
    
    conversation_log = ""
    total_tokens_used = 0
    off_topic_count = 0
    question_count = 0

    # Initialize conversation history with the lecture summary (not the full raw transcript)
    messages = [
        {"role": "system", "content": (
            "You are a professional interviewer assessing a user's understanding of a lecture. "
            "Ask natural follow-up questions based on the user's responses. Avoid referencing raw transcript text. "
            "Focus on the key topics in the lecture summary provided. "
            "If the user gives off-topic or irrelevant responses, issue a polite warning. "
            "After three off-topic responses, ask a targeted question from a topic not yet addressed. "
            "If the user still does not provide a relevant answer, conclude the interview."
        )},
        {"role": "user", "content": f"Lecture Summary:\n{summary_text}"}
    ]

    while question_count < 10:
        # If off_topic_count is below 3, generate a normal question.
        if off_topic_count < 3:
            question, tokens_used = get_next_question(messages)
        else:
            # Use a targeted question prompt if there have been 3 off-topic responses.
            remaining_topics = [t for t in topics if t not in asked_topics]
            if remaining_topics:
                special_prompt = (
                    "Let's refocus on the lecture content. Please ask a question specifically related to one of these key topics: "
                    + ", ".join(remaining_topics) +
                    ". Provide a question that requires a relevant answer."
                )
            else:
                special_prompt = "Let's continue with relevant questions."
            messages.append({"role": "user", "content": special_prompt})
            question, tokens_used = get_next_question(messages)
        total_tokens_used += tokens_used

        conversation_log += f"Question {question_count + 1}: {question}\n"
        store_question(question)
        print(f"\n{question}")

        # Get user's answer
        user_response = input("Answer: ").strip()
        conversation_log += f"Answer: {user_response}\n"
        store_user_response(user_response)

        # Append Q/A to the conversation history
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": user_response})

        # Check if the response is off-topic.
        if is_off_topic(user_response):
            off_topic_count += 1
            warning = f"⚠️ Warning {off_topic_count}/3: Your response is off-topic. Please focus on the lecture content."
            print(warning)
            conversation_log += warning + "\n"

            # If this is the third off-topic response, ask a targeted question to bring the focus back.
            if off_topic_count == 3:
                remaining_topics = [t for t in topics if t not in asked_topics]
                if remaining_topics:
                    special_prompt = (
                        "We need to refocus on the lecture content. Please answer this question about one of these topics: "
                        + ", ".join(remaining_topics) +
                        "."
                    )
                else:
                    special_prompt = "Let's continue with a relevant question."
                messages.append({"role": "user", "content": special_prompt})
                question_targeted, tokens_used = get_next_question(messages)
                total_tokens_used += tokens_used
                print(f"\nRefocus Question: {question_targeted}")
                conversation_log += f"Refocus Question: {question_targeted}\n"
                store_question(question_targeted)
                user_response = input("Your answer: ").strip()
                conversation_log += f"Answer: {user_response}\n"
                store_user_response(user_response)
                messages.append({"role": "assistant", "content": question_targeted})
                messages.append({"role": "user", "content": user_response})
                # If user still goes off-topic on refocus, then terminate.
                if is_off_topic(user_response):
                    final_msg = "Your answers remain off-topic. The interview will now conclude. Thank you."
                    print(final_msg)
                    conversation_log += final_msg + "\n"
                    break
                else:
                    # Reset off_topic_count if the user gives a relevant answer.
                    off_topic_count = 0
        else:
            # Mark any matching topics as addressed (very basic keyword check)
            for topic in topics:
                if topic.lower() in user_response.lower():
                    asked_topics.add(topic)
        
        question_count += 1

    store_conversation(conversation_log)
    print("\n✅ Interview finished. All data stored.")
    print(f"🧠 Total tokens used: {total_tokens_used}")

if __name__ == "__main__":
    interview()
