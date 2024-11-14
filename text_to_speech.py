import os
import threading
from gtts import gTTS

def play_text_to_speech(text, request_id, language='en-IN', slow=False):
    """Generate TTS audio and save it as an MP3 file."""
    def tts_and_save():
        # Generate TTS audio
        tts = gTTS(text=text, lang=language, slow=slow)
        temp_audio_file = f"response_{request_id}.mp3"
        tts.save(temp_audio_file)

    # Run the TTS save function in a separate thread
    threading.Thread(target=tts_and_save).start()
