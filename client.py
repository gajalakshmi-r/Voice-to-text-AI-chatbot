# client.py
import asyncio
import websockets
import pyaudio

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

async def audio_sender(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    try:
        while True:
            audio_data = stream.read(CHUNK)
            await websocket.send(audio_data)
    except Exception as e:
        print(f"Error sending audio: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def receive_messages(websocket):
    try:
        while True:
            message = await websocket.recv()
            print(f"Received message: {message}")
    except Exception as e:
        print(f"Error receiving message: {e}")

async def main():
    uri = "ws://127.0.0.1:8000/ws/test_id"
    async with websockets.connect(uri) as websocket:
        await asyncio.gather(
            audio_sender(websocket),
            receive_messages(websocket)
        )

if __name__ == "__main__":
    asyncio.run(main())
