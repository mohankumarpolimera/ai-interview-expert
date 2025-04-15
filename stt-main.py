import openai
import pymongo
import pyodbc
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import time
import threading
import asyncio

# Choose one of these TTS options:
# Option 1: Pyttsx3 - Fast but lower quality
USE_PYTTSX3 = False
if USE_PYTTSX3:
    import pyttsx3
    
# Option 2: gTTS - Google's TTS (requires internet)
USE_GTTS = False 
if USE_GTTS:
    from gtts import gTTS
    import io
    from pydub import AudioSegment
    import simpleaudio as sa

# Option 3: Edge-TTS - Microsoft Edge's TTS (requires internet but fast streaming)
USE_EDGE_TTS = True
if USE_EDGE_TTS:
    import edge_tts
    from pydub import AudioSegment
    import simpleaudio as sa

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

# Initialize TTS engine if using pyttsx3
if USE_PYTTSX3:
    tts_engine = pyttsx3.init()
    # Speed up the rate (default is 200)
    tts_engine.setProperty('rate', 225)
    # Get available voices and use a better one if available
    voices = tts_engine.getProperty('voices')
    # Try to find a female voice
    for voice in voices:
        if "female" in voice.name.lower():
            tts_engine.setProperty('voice', voice.id)
            break

# Audio parameters for recording
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1         # Mono
BLOCK_SIZE = 4096    # Larger for faster processing
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0

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
# Speech-to-Text Functions
# ------------------------

def record_audio():
    """Record audio until silence is detected or max time reached"""
    print("Listening... (speak now)")
    
    audio_chunks = []
    silence_start = None
    recording_start = time.time()
    recording = True
    
    # Increase silence duration to 3 seconds
    SILENCE_DURATION = 3.0  # Stop after 3 seconds of silence
    MAX_RECORDING_TIME = 30.0  # Maximum recording time in seconds
    
    def audio_callback(indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata**2))
        audio_chunks.append(indata.copy())
        
        nonlocal silence_start, recording, recording_start
        
        # Check if max recording time has been reached
        if time.time() - recording_start > MAX_RECORDING_TIME:
            recording = False
            print("Max recording time reached.")
            raise sd.CallbackStop()
            
        # Check for silence
        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
                print("Silence detected, waiting to confirm...")
            elif time.time() - silence_start > SILENCE_DURATION:
                recording = False
                print("Silence confirmed. Stopping recording.")
                raise sd.CallbackStop()
        else:
            if silence_start is not None:
                print("Silence broken, continuing to record.")
            silence_start = None

    try:
        # Try to use default microphone
        input_device = None  # Use default microphone
        
        # Display a countdown timer for the user
        print(f"Recording will automatically stop after {MAX_RECORDING_TIME} seconds or 3 seconds of silence")
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            callback=audio_callback,
            device=input_device
        ):
            while recording:
                sd.sleep(100)
                elapsed = time.time() - recording_start
                if elapsed % 5 < 0.1:  # Show time remaining every 5 seconds
                    remaining = MAX_RECORDING_TIME - elapsed
                    if remaining > 0:
                        print(f"Recording: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
    except Exception as e:
        print(f"Recording error: {e}")
        return None

    if not audio_chunks:
        return None
            
    # Concatenate audio chunks
    audio = np.concatenate(audio_chunks, axis=0)
    if len(audio) / SAMPLE_RATE < 0.5:  # Discard if shorter than 0.5s
        print("Response too short, discarding.")
        return None
    
    print(f"Recording complete. Duration: {len(audio) / SAMPLE_RATE:.1f} seconds")
    
    # Save to temp file
    temp_file = "temp_input.wav"
    wavfile.write(temp_file, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return temp_file

def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    try:
        start_time = time.time()
        with open(audio_file, "rb") as file:
            transcription = openai.Audio.transcribe(
                "whisper-1",
                file,
                language="en"
            )
        print(f"Transcription time: {time.time() - start_time:.2f}s")
        return transcription["text"].strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

# ------------------------
# Text-to-Speech Functions
# ------------------------

def text_to_speech_pyttsx3(text):
    """Convert text to speech using pyttsx3 (fast local TTS)"""
    start_time = time.time()
    tts_engine.say(text)
    tts_engine.runAndWait()
    print(f"TTS time: {time.time() - start_time:.2f}s")

def text_to_speech_gtts(text):
    """Convert text to speech using Google TTS"""
    start_time = time.time()
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        temp_file = "temp_output.mp3"
        tts.save(temp_file)

        # Play using pydub + simpleaudio
        sound = AudioSegment.from_file(temp_file, format="mp3")
        play_obj = sa.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        play_obj.wait_done()

        os.remove(temp_file)
        print(f"TTS time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"TTS error: {e}")

async def text_to_speech_edge(text):
    """Convert text to speech using Microsoft Edge TTS (streaming)"""
    start_time = time.time()
    try:
        # Use a fast voice - you can change this to other voices
        voice = "en-US-AriaNeural"
        
        # Create communication
        communicate = edge_tts.Communicate(text, voice)
        
        # Save to file and play
        temp_file = "temp_output.mp3"
        await communicate.save(temp_file)
        
        # Play using pydub + simpleaudio
        sound = AudioSegment.from_file(temp_file, format="mp3")
        play_obj = sa.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        play_obj.wait_done()
        
        # Clean up
        os.remove(temp_file)
        print(f"TTS time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Edge TTS error: {e}")

def text_to_speech(text):
    """Convert text to speech using the selected method"""
    if USE_PYTTSX3:
        text_to_speech_pyttsx3(text)
    elif USE_GTTS:
        text_to_speech_gtts(text)
    elif USE_EDGE_TTS:
        asyncio.run(text_to_speech_edge(text))

# ------------------------
# Main Interview Loop
# ------------------------

def interview():
    print("=== Voice-Based Interview System Started ===")
    print("Press Ctrl+C to exit at any time")
    
    # Fetch and summarize the lecture transcript
    print("Loading lecture data...")
    transcript = fetch_transcript()
    summary_text, topics = generate_summary(transcript)
    print("Lecture data loaded successfully!")
    
    # For tracking topics already covered
    asked_topics = set()
    
    # For storing the complete conversation
    conversation_log = ""
    total_tokens_used = 0
    off_topic_count = 0
    question_count = 0

    # Initialize conversation history with the lecture summary (not the full raw transcript)
    messages = [
        {"role": "system", "content": (
            "You are a professional interviewer assessing a user's understanding of a lecture. "
            "Ask natural and concise follow-up questions based on the user's responses and questions based on lecture mainly."
            "Avoid referencing raw transcript text. "
            "Focus on the key topics in the lecture summary provided. "
            "If the user gives off-topic or irrelevant responses, issue a polite warning. "
            "After three off-topic responses, ask a targeted question from a topic not yet addressed. "
            "If the user still does not provide a relevant answer, conclude the interview."
        )},
        {"role": "user", "content": f"Lecture Summary:\n{summary_text}"}
    ]

    # Welcome message
    welcome_message = "Welcome to the voice-based interview system. I'll ask you questions about the lecture material. Please respond clearly when you hear the listening prompt."
    print(welcome_message)
    text_to_speech(welcome_message)
    
    try:
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
            print(f"\nInterviewer: {question}")
            
            # Convert question to speech
            text_to_speech(question)

            # Get user's answer using speech-to-text
            audio_file = record_audio()
            if not audio_file:
                retry_message = "I didn't catch that. Could you please speak again?"
                print(f"System: {retry_message}")
                text_to_speech(retry_message)
                continue
                
            user_response = transcribe_audio(audio_file)
            os.remove(audio_file)  # Clean up the temporary audio file
            
            if not user_response:
                retry_message = "I couldn't understand what you said. Let's try again."
                print(f"System: {retry_message}")
                text_to_speech(retry_message)
                continue
                
            print(f"You: {user_response}")
            conversation_log += f"Answer: {user_response}\n"
            store_user_response(user_response)

            # Append Q/A to the conversation history
            messages.append({"role": "assistant", "content": question})
            messages.append({"role": "user", "content": user_response})

            # Check if the response is off-topic.
            if is_off_topic(user_response):
                off_topic_count += 1
                warning = f"Warning {off_topic_count}/3: Your response is off-topic. Please focus on the lecture content."
                
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
                    print(f"\nInterviewer (Refocus): {question_targeted}")
                    conversation_log += f"Refocus Question: {question_targeted}\n"
                    store_question(question_targeted)
                    
                    # Convert refocus question to speech
                    text_to_speech(question_targeted)
                    
                    # Get user's answer using speech-to-text
                    audio_file = record_audio()
                    if not audio_file:
                        continue
                        
                    user_response = transcribe_audio(audio_file)
                    os.remove(audio_file)
                    
                    if not user_response:
                        continue
                        
                    print(f"You: {user_response}")
                    conversation_log += f"Answer: {user_response}\n"
                    store_user_response(user_response)
                    messages.append({"role": "assistant", "content": question_targeted})
                    messages.append({"role": "user", "content": user_response})
                    
                    # If user still goes off-topic on refocus, then terminate.
                    if is_off_topic(user_response):
                        final_msg = "Your answers remain off-topic. The interview will now conclude. Thank you."
                        print(f"System: {final_msg}")
                        conversation_log += final_msg + "\n"
                        text_to_speech(final_msg)
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

        # Conclude the interview
        conclusion_message = f"Interview completed. Thank you for your participation. {question_count} questions were asked."
        print(f"\nSystem: {conclusion_message}")
        text_to_speech(conclusion_message)
        
        # Store conversation in database
        store_conversation(conversation_log)
        print("\n✅ Interview finished. All data stored.")
        print(f"🧠 Total tokens used: {total_tokens_used}")
        
    except KeyboardInterrupt:
        print("\n=== Interview System Stopped by User ===")
        # Still store partial conversation
        store_conversation(conversation_log)
        print("Partial conversation data stored.")

if __name__ == "__main__":
    interview()