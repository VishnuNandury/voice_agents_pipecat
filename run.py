#
# Quick-start runner for the Loan Collection Voice Agent
#
# Usage:
#   python run.py                  # Multi-pipeline dashboard (local)
#   python run.py --single         # Single pipeline via PipeCat runner (local)
#   python run.py --cloud          # Cloud-ready server with TURN support
#

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv(override=True)


def check_env():
    """Validate that required API keys are present."""
    required = {
        "DEEPGRAM_API_KEY": "https://console.deepgram.com/signup",
        "OPENAI_API_KEY": "https://platform.openai.com/signup",
    }
    missing = []
    for key, url in required.items():
        val = os.getenv(key, "").strip()
        if not val or val.startswith("your_"):
            missing.append(f"  {key}  ->  Get it at: {url}")

    if missing:
        print("\n[ERROR] Missing API keys in your .env file:\n")
        for m in missing:
            print(m)
        print("\nSteps:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API keys")
        print("  3. Run this script again\n")
        sys.exit(1)

    print("[OK] All API keys found")


def main():
    print("=" * 55)
    print("  QuickFinance - Loan Collection Voice Agent")
    print("  Hindi | English | Hinglish")
    print("=" * 55)
    print()

    check_env()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cloud_mode = "--cloud" in sys.argv
    single_mode = "--single" in sys.argv

    if cloud_mode:
        turn = os.getenv("TURN_URL", "").strip()
        print(f"\n[CLOUD MODE] TURN: {'configured' if turn else 'NOT SET (audio may fail)'}")
        print("[STARTING] Launching cloud-ready server with TURN support...")
        print("[INFO] Open http://localhost:7860/client in your browser\n")
        cmd = [sys.executable, "app.py"]

    elif single_mode:
        stt = os.getenv("PIPELINE_STT", "deepgram")
        tts = os.getenv("PIPELINE_TTS", "openai")
        print(f"\n[SINGLE MODE] STT={stt} | TTS={tts}")
        print("[STARTING] Launching bot with WebRTC transport...")
        print("[INFO] Open http://localhost:7860/client in your browser\n")
        cmd = [sys.executable, "bot.py", "-t", "webrtc"]

    else:
        print("\n[MULTI-PIPELINE MODE] Compare 4 STT+TTS combinations")
        print("[STARTING] Launching dashboard server...")
        print("[INFO] Open http://localhost:7860 in your browser")
        print("[INFO] Click Connect on any pipeline card, then Open Voice UI\n")
        cmd = [sys.executable, "server.py"]

    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n[STOPPED] Agent shut down gracefully.")
    except FileNotFoundError:
        print("\n[ERROR] Could not find required files.")
        print("Make sure you installed dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
