from __future__ import annotations

import base64
import io
import os
import tempfile
import time

import cv2
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from litellm import completion, transcription

# --- Config ---
# Keep these short and readable so the loop is easy to tweak.
AUDIO_SECONDS = 4
AUDIO_SAMPLE_RATE = 16_000

# Gemini models through LiteLLM.
TRANSCRIPTION_MODEL = "gemini/gemini-2.0-flash"
VISION_TEXT_MODEL = "gemini/gemini-2.0-flash"

# ElevenLabs voice settings.
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George" default voice


GOOFY_SYSTEM_PROMPT = """
Eres una cámara de seguridad IA muy graciosa y exagerada.
Habla SIEMPRE en español.
Sé breve (1-3 frases), divertida y amigable.
Describe qué ves y qué escuchas, pero evita inventar hechos peligrosos.
""".strip()


def capture_webcam_frame() -> str:
    """Capture one frame from the default webcam and return it as base64 JPEG."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (índice 0).")

    ok, frame = cam.read()
    cam.release()

    if not ok:
        raise RuntimeError("No se pudo capturar un frame de la webcam.")

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen como JPEG.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def record_microphone_wav(seconds: int = AUDIO_SECONDS) -> str:
    """Record microphone audio to a temporary WAV file and return its path."""
    print(f"🎤 Grabando audio por {seconds} segundos...")
    recording = sd.rec(
        int(seconds * AUDIO_SAMPLE_RATE),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, recording, AUDIO_SAMPLE_RATE)
        return tmp.name


def transcribe_audio_with_gemini(wav_path: str) -> str:
    """Transcribe WAV audio using Gemini via LiteLLM transcription endpoint."""
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "mic.wav"  # Some providers use this metadata.

    result = transcription(
        model=TRANSCRIPTION_MODEL,
        file=audio_file,
    )
    return result.text.strip()


def generate_goofy_reply(frame_b64: str, transcript_text: str) -> str:
    """Use a multimodal Gemini prompt (image + transcript) to craft the response."""
    user_content = [
        {
            "type": "text",
            "text": (
                "Este es el contexto del micrófono transcrito: "
                f"'{transcript_text or '[sin audio claro]'}'. "
                "Describe qué está pasando como una cámara de seguridad cómica."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        },
    ]

    response = completion(
        model=VISION_TEXT_MODEL,
        messages=[
            {"role": "system", "content": GOOFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.9,
        max_tokens=100,
    )

    return response.choices[0].message.content.strip()


def speak_with_elevenlabs(text: str, voice_id: str) -> None:
    """Convert text to speech with ElevenLabs and play it locally."""
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        text=text,
    )

    # Save + play using sounddevice/soundfile (simple, no external player required).
    # The ElevenLabs SDK returns an iterator of audio chunks.
    audio_bytes = b"".join(audio)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        mp3_path = tmp.name

    data, sr = sf.read(mp3_path, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def main() -> None:
    load_dotenv()

    # Required API keys:
    # - GEMINI_API_KEY for LiteLLM Gemini calls
    # - ELEVENLABS_API_KEY for voice synthesis
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Falta GEMINI_API_KEY en variables de entorno.")
    if not os.getenv("ELEVENLABS_API_KEY"):
        raise RuntimeError("Falta ELEVENLABS_API_KEY en variables de entorno.")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
    loop_delay = float(os.getenv("LOOP_DELAY_SECONDS", "1.0"))

    print("🤖 Cámara IA graciosa iniciada. Ctrl+C para salir.")
    while True:
        try:
            frame_b64 = capture_webcam_frame()
            wav_path = record_microphone_wav()
            transcript_text = transcribe_audio_with_gemini(wav_path)

            print(f"📝 Transcripción: {transcript_text or '[vacío]'}")
            reply = generate_goofy_reply(frame_b64, transcript_text)
            print(f"🗣️ Respuesta IA: {reply}")

            speak_with_elevenlabs(reply, voice_id)
            time.sleep(loop_delay)
        except KeyboardInterrupt:
            print("\n👋 Saliendo...")
            break
        except Exception as e:
            print(f"⚠️ Error en loop: {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
