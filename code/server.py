from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import openai
import asyncio
import websockets
import json
from dotenv import load_dotenv
import os

# Load environment variables for OpenAI API keys
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# WebSocket endpoint for audio processing
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    print(f"New connection with user_id: {user_id}")
    await websocket.accept()
    try:
        while True:
            # Receive the audio data from the client
            audio_data = await websocket.receive_bytes()
            print(f"Received {len(audio_data)} bytes of audio data from user {user_id}")
            
            # Call OpenAI's Whisper API to transcribe the audio (this is where you process the audio)
            transcription = await transcribe_audio_with_whisper(audio_data)
            
            # Send the transcription result to GPT-4 for response
            gpt_response = await get_gpt_response(transcription)
            
            # Send the GPT-4 response back to the client
            await websocket.send_text(gpt_response)
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")

async def transcribe_audio_with_whisper(audio_data: bytes):
    # Call the OpenAI Whisper API (or other service) to transcribe the audio data
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_data
    )
    return response['text']

async def get_gpt_response(transcription: str):
    # Send the transcription to GPT-4 for a response
    response = openai.Completion.create(
        model="gpt-4",
        prompt=transcription,
        max_tokens=150
    )
    return response['choices'][0]['text']
