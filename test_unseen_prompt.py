"""A fixed inference prompt deliberately not included in the downloaded source corpus."""
import subprocess, sys
prompt = "A curious student asks: How can people learn from mistakes and improve their daily habits?\nAnswer:"
cmd = [sys.executable, "chat.py", "--checkpoint", "runs/best.pt",
       "--prompt", prompt, "--max-new-tokens", "350"]
result = subprocess.run(cmd, text=True)
if result.returncode == 0:
    from pathlib import Path
    Path("runs/generated_unseen_prompt.txt").write_text(
        "PROMPT (inference-only):\n" + prompt + "\n\n" +
        "Run output is printed above. Re-run chat.py to regenerate.\n",
        encoding="utf-8"
    )
sys.exit(result.returncode)
