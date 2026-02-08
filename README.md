# Loan Collection Voice Agent

A **demo-ready, voice-to-voice** conversational AI agent for loan collection, built with [PipeCat](https://github.com/pipecat-ai/pipecat). The agent ("Priya") naturally speaks **Hindi, English, and Hinglish** (code-switching) based on the customer's language.

---

## Features

- **Real-time voice conversation** - speak and listen naturally via your browser
- **Trilingual support** - Hindi, English, and seamless Hinglish mixing
- **Realistic loan collection flow** - greeting, verification, empathetic negotiation, payment options, commitment
- **Interruptible** - the customer can interrupt the agent mid-sentence (like a real call)
- **Empathetic tone** - handles difficult situations (angry customers, inability to pay) gracefully
- **Browser-based demo** - no phone line needed, just open a browser tab

## Architecture

```
Browser (Mic/Speaker)
    |
    |  WebRTC audio stream
    v
PipeCat Pipeline:
    [Mic Input] --> [Deepgram STT (Nova-3, multilingual)]
                        --> [OpenAI GPT-4o (Loan Collection Agent)]
                              --> [OpenAI TTS (Hindi/English voice)]
                                    --> [Speaker Output]
```

| Component | Service | Purpose |
|-----------|---------|---------|
| STT | Deepgram Nova-3 (`language=multi`) | Hindi + English + Hinglish speech recognition |
| LLM | OpenAI GPT-4o | Conversational intelligence with bilingual system prompt |
| TTS | OpenAI TTS | Natural Hindi/English speech synthesis |
| Transport | PipeCat WebRTC | Real-time browser audio streaming |

---

## Quick Start

### Prerequisites

- **Python 3.10+** installed
- A working **microphone** and **speakers/headphones**
- API keys (free tiers available):
  - [Deepgram](https://console.deepgram.com/signup) - $200 free credit, no credit card
  - [OpenAI](https://platform.openai.com/signup) - pay-as-you-go

### Step 1: Install Dependencies

```bash
cd pipecat

# Create a virtual environment (recommended)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your real API keys:
#   DEEPGRAM_API_KEY=your_actual_deepgram_key
#   OPENAI_API_KEY=your_actual_openai_key
```

Open `.env` in any text editor and replace the placeholder values with your actual API keys.

### Step 3: Run the Agent

**Option A: Using the quick-start runner (recommended)**

```bash
python run.py
```

This validates your setup and launches the bot with PipeCat's WebRTC transport. A browser window opens automatically.

**Option B: Using PipeCat runner directly**

```bash
python bot.py -t webrtc
```

### Step 4: Start Talking

1. Open **http://localhost:7860/client** in your browser (Chrome recommended)
2. Click the **Connect** button
3. **Allow microphone access** when your browser asks
4. Wait for Priya to greet you
5. **Respond in Hindi, English, or Hinglish** - the agent adapts automatically
6. Click **Disconnect** to end the call

---

## Demo Scenario

The agent comes preloaded with this demo scenario:

| Field | Value |
|-------|-------|
| Customer Name | Rajesh Kumar |
| Account Number | QF-2024-78432 |
| Loan Type | Personal Loan |
| Outstanding Amount | Rs. 47,500 |
| Monthly EMI | Rs. 5,280 |
| Overdue EMIs | 2 months (Dec 2025, Jan 2026) |
| Late Fee | Rs. 1,200 |
| Total Due Now | Rs. 11,760 |

**Try these conversation flows:**

1. **Cooperative customer**: Confirm identity, acknowledge the dues, agree to pay
2. **Difficulty paying**: Say you lost your job / had medical expenses - agent offers restructuring
3. **Angry customer**: Express frustration - agent stays calm and empathetic
4. **Hinglish conversation**: Mix Hindi and English freely - "Haan, mujhe pata hai about the EMI, but I had some problems"

---

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | (required) | Deepgram API key for speech recognition |
| `OPENAI_API_KEY` | (required) | OpenAI API key for LLM + TTS |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model for conversation |
| `OPENAI_TTS_VOICE` | `alloy` | TTS voice (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram STT model |
| `DEEPGRAM_LANGUAGE` | `multi` | STT language (`multi` for auto-detect, `hi` for Hindi-only, `en` for English-only) |
| `SERVER_PORT` | `7860` | Port for standalone server |

---

## Project Structure

```
pipecat/
  bot.py              # Core agent: pipeline + system prompt + loan collection logic
  run.py              # Quick-start launcher with validation
  server.py           # Standalone FastAPI server (alternative deployment)
  requirements.txt    # Python dependencies
  .env.example        # Template for API keys
  .env                # Your actual API keys (not committed to git)
  static/
    index.html        # Browser UI for voice interaction
  README.md           # This file
```

---

## Customization

### Change the loan details

Edit the `SYSTEM_PROMPT` in `bot.py` - look for the "Loan Details" section and update the amounts, dates, and customer name.

### Change the agent persona

Edit the name, company, and behavior rules in the `SYSTEM_PROMPT` at the top of `bot.py`.

### Add more languages

Change `DEEPGRAM_LANGUAGE` in `.env` to support different language combinations. Deepgram Nova-3 supports 30+ languages.

### Switch TTS voice

Set `OPENAI_TTS_VOICE` in `.env`. Try `nova` for a different female voice or `onyx` for a male voice.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Missing API keys" error | Make sure `.env` exists and has real keys (not placeholder values) |
| No audio from agent | Check browser audio permissions, try Chrome |
| Agent doesn't understand Hindi | Ensure `DEEPGRAM_LANGUAGE=multi` in `.env` |
| High latency | Use a wired internet connection; GPT-4o-mini is faster but less capable |
| Microphone not detected | Check OS microphone permissions for your browser |
| Import errors | Make sure you activated the virtual environment and ran `pip install -r requirements.txt` |
