import os
import sys
import time
import httpx

TARGET_DIR = r"D:\localllamacpp\models\gemma4-12b\bf16"
TARGET_FILE = os.path.join(TARGET_DIR, "model.safetensors")
INCOMPLETE_FILE = TARGET_FILE + ".part"
EXPECTED_SIZE = 23919549408  # exact bytes

URL = "https://huggingface.co/google/gemma-4-12B-it/resolve/main/model.safetensors"

def format_time(seconds):
    if seconds is None or seconds < 0 or seconds > 86400 * 7:
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

def download():
    os.makedirs(TARGET_DIR, exist_ok=True)
    if os.path.exists(TARGET_FILE) and os.path.getsize(TARGET_FILE) == EXPECTED_SIZE:
        print(f"[Done] {TARGET_FILE} already fully downloaded ({EXPECTED_SIZE / 1e9:.2f} GB).", flush=True)
        return

    downloaded = os.path.getsize(INCOMPLETE_FILE) if os.path.exists(INCOMPLETE_FILE) else 0
    print(f"Resilient Download: model.safetensors", flush=True)
    print(f"Target: {TARGET_FILE}", flush=True)
    print(f"Expected Size: {EXPECTED_SIZE / 1e9:.3f} GB | Already present: {downloaded / 1e9:.3f} GB", flush=True)

    max_retries = 100
    retry = 0

    while downloaded < EXPECTED_SIZE and retry < max_retries:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Antigravity/1.0",
        }
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=30.0)) as client:
                with client.stream("GET", URL, headers=headers) as response:
                    if response.status_code not in (200, 206):
                        print(f"HTTP Error {response.status_code}. Retrying in 5s...", flush=True)
                        time.sleep(5)
                        retry += 1
                        continue

                    mode = "ab" if downloaded > 0 else "wb"
                    last_log_time = time.time()
                    last_log_bytes = downloaded
                    start_time = time.time()
                    bytes_this_session = 0

                    with open(INCOMPLETE_FILE, mode) as f:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            bytes_this_session += len(chunk)

                            now = time.time()
                            if now - last_log_time >= 15.0:
                                elapsed_window = now - last_log_time
                                speed_window = (downloaded - last_log_bytes) / elapsed_window  # B/s
                                speed_mb = speed_window / (1024 * 1024)

                                remaining_bytes = EXPECTED_SIZE - downloaded
                                eta_s = remaining_bytes / speed_window if speed_window > 0 else 0
                                pct = (downloaded / EXPECTED_SIZE) * 100.0

                                print(
                                    f"[{time.strftime('%H:%M:%S')}] {downloaded / 1e9:6.2f} / {EXPECTED_SIZE / 1e9:.2f} GB "
                                    f"({pct:5.1f}%) | Speed: {speed_mb:5.2f} MB/s | ETA: {format_time(eta_s)}",
                                    flush=True
                                )
                                last_log_time = now
                                last_log_bytes = downloaded

        except Exception as e:
            print(f"Network interruption ({e}). Reconnecting in 5s...", flush=True)
            time.sleep(5)
            retry += 1
            if os.path.exists(INCOMPLETE_FILE):
                downloaded = os.path.getsize(INCOMPLETE_FILE)

    if downloaded >= EXPECTED_SIZE:
        if os.path.exists(TARGET_FILE):
            os.remove(TARGET_FILE)
        os.rename(INCOMPLETE_FILE, TARGET_FILE)
        print(f"[SUCCESS] Download completed! Verified size: {os.path.getsize(TARGET_FILE)} bytes.", flush=True)
    else:
        print(f"[FAILED] Download did not complete after {max_retries} attempts.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    download()
