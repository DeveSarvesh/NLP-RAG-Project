"""
Quick Gemini API diagnostic - run this to see the exact error.
"""
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GEMINI_API_KEY", "NOT SET")
print(f"Key prefix : {key[:12]}...")
print(f"Key length : {len(key)} chars")
print()

try:
    from google import genai
    import importlib.metadata
    print(f"google-genai SDK version: {importlib.metadata.version('google-genai')}")
except Exception as e:
    print(f"Import error: {e}")
    raise SystemExit(1)

client = genai.Client(api_key=key)
print("Client initialised OK\n")

for model in ["gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
    print(f"Trying {model} ...", end=" ", flush=True)
    try:
        r = client.models.generate_content(model=model, contents="Reply with just: WORKING")
        print(f"SUCCESS -> {r.text.strip()}")
        break
    except Exception as e:
        print(f"FAILED -> {type(e).__name__}: {str(e)[:180]}")

print("\nDiagnostic complete.")
