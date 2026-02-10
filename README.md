# AI Camera Speaker Module (simple)

Mini loop de **"cámara de seguridad IA graciosa"** que:

1. Toma un frame de webcam.
2. Graba unos segundos del micrófono.
3. Transcribe el audio con **Gemini a través de LiteLLM**.
4. Genera una respuesta corta, divertida y en español (usando imagen + texto).
5. Convierte esa respuesta a voz con **ElevenLabs** y la reproduce.

## Requisitos

- Python 3.11+
- Webcam y micrófono funcionando
- API keys:
  - `GEMINI_API_KEY`
  - `ELEVENLABS_API_KEY`

## Instalación (con uv)

```bash
uv sync
cp .env.example .env
# edita .env con tus claves
```

## Ejecutar

```bash
uv run python main.py
```

## Notas

- El código está hecho para ser **simple y fácil de leer**.
- Puedes cambiar la personalidad editando `GOOFY_SYSTEM_PROMPT` en `main.py`.
- Si tu dispositivo de audio es distinto al predeterminado, configura el default de `sounddevice` en tu sistema.
