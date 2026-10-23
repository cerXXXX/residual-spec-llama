import urllib.request
import json
import time

# Create a ~20k token prompt (repeating realistic text)
text = """
def process_data_pipeline(data: list[dict], threshold: float = 0.5) -> dict:
    '''Process incoming telemetry data and compute statistics.'''
    result = {'valid': 0, 'invalid': 0, 'metrics': []}
    for item in data:
        score = item.get('score', 0.0)
        if score >= threshold:
            result['valid'] += 1
            result['metrics'].append(score * 1.5)
        else:
            result['invalid'] += 1
    return result

""" * 400 # ~20k tokens

req_data = {
    "model": "gemma-4-E4B-it-Q4_K_M",
    "messages": [
        {"role": "user", "content": f"Here is code documentation:\n{text}\n\nSummarize what the code does in 3 bullet points."}
    ],
    "max_tokens": 200,
    "temperature": 0.0,
    "stream": False
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer secret-test-key"
}

print(f"Sending request with ~{len(text)//4} tokens prompt...")
t0 = time.perf_counter()
req = urllib.request.Request("http://127.0.0.1:9095/v1/chat/completions", data=json.dumps(req_data).encode("utf-8"), headers=headers)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    print(f"Prompt tokens: {pt}, Completion tokens: {ct}, Total time: {elapsed:.2f}s")
    print(f"Content: {data['choices'][0]['message']['content']}")
