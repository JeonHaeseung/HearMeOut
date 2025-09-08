import json
import numpy as np

def levenshtein(a, b):
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = np.zeros((la+1, lb+1), dtype=int)
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[la][lb]

def word_similarity(a, b):
    max_len = max(len(a), len(b))
    if max_len == 0: return 1.0
    return 1.0 - levenshtein(a, b)/max_len

def align_transcript(prompt, transcript, mask_token="<|startoflm|>"):
    prompt_words = prompt.split()
    transcript_words = transcript.split()
    
    if len(transcript_words) >= len(prompt_words):
        return transcript
    
    aligned = [mask_token] * len(prompt_words)  
    
    used_idx = set()
    for w in transcript_words:
        best_sim = -1
        best_idx = -1
        for i, pw in enumerate(prompt_words):
            if i in used_idx:
                continue
            sim = word_similarity(w, pw)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0:
            aligned[best_idx] = w
            used_idx.add(best_idx)
    
    return " ".join(aligned)

def process_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        prompt = item.get("prompt", "")
        transcript = item.get("response", "")
        item["response"] = align_transcript(prompt, transcript)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_json("input_file_path", "output_file_path")
