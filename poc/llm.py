"""All model calls go through llm() and embed(). Provider and model come from config/.env."""
import hashlib
import json
import os
import re
import time

import httpx

import config

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"
RETRY_STATUS = (429, 500, 502, 503, 504)


class Overloaded(Exception):
    pass


class QuotaExhausted(Exception):
    pass


_exhausted = set()  # models whose quota ran out during this run (free tier: 20 requests/day/model)
MAX_WAIT = 90  # a 429 asking to wait longer than this is a daily quota, not a per-minute limit


def _gemini(model, method, body, tries=8, max_overload=None):
    """POST to the Gemini API, retrying rate limits (429, honouring retryDelay) and overload (503).
    After `max_overload` 503s raises Overloaded so the caller can switch model."""
    headers = {"X-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}
    url = GEMINI_URL.format(model=model, method=method)
    overloads = 0
    for attempt in range(tries):
        r = httpx.post(url, headers=headers, json=body, timeout=180)
        if r.status_code == 503:
            overloads += 1
            if max_overload and overloads >= max_overload:
                raise Overloaded(model)
        if r.status_code not in RETRY_STATUS or attempt == tries - 1:
            r.raise_for_status()
            return r.json()
        m = re.search(r'"retryDelay":\s*"(\d+)', r.text)
        wait = int(m.group(1)) + 1 if m else min(60, 2 ** attempt * 2)
        if r.status_code == 429 and wait > MAX_WAIT:
            _exhausted.add(model)
            raise QuotaExhausted(f"{model}: quota exhausted (retry in {wait}s)")
        print(f"  gemini {model} {r.status_code}; retrying in {wait}s")
        time.sleep(wait)


def llm(prompt, system=None, json_mode=False, judge=False, temperature=0.0):
    """One text completion. judge=True uses the separately configured judge model.
    Responses are cached on disk by (role, provider, model, system, prompt, json_mode)."""
    provider = config.JUDGE_PROVIDER if judge else config.LLM_PROVIDER
    model = config.JUDGE_MODEL if judge else config.LLM_MODEL
    key = hashlib.sha256(json.dumps([judge, provider, model, system, prompt, json_mode]).encode()).hexdigest()
    path = config.CACHE_DIR / f"{key}.json"
    if config.LLM_CACHE and path.exists():
        return json.loads(path.read_text())["text"]
    text = _complete(provider, model, prompt, system, json_mode, temperature)
    if config.LLM_CACHE and text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}))
    return text


def _complete(provider, model, prompt, system, json_mode, temperature):
    if provider == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        models = [m for m in [model] + config.LLM_FALLBACK_MODELS if m not in _exhausted]
        models = list(dict.fromkeys(models))
        if not models:
            raise QuotaExhausted("all configured Gemini models are out of quota")
        for i, m in enumerate(models):
            last = i == len(models) - 1
            try:  # overloaded or out-of-quota model -> next one; the last one keeps retrying overloads
                resp = _gemini(m, "generateContent", body, max_overload=None if last else 3)
                break
            except (Overloaded, QuotaExhausted) as e:
                if last:
                    raise
                print(f"  {e if isinstance(e, QuotaExhausted) else m + ' overloaded'}; falling back to {models[i + 1]}")
        parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if provider == "anthropic":
        return _claude(model, prompt, system, json_mode)
    if provider == "openai":
        from openai import OpenAI

        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = OpenAI().chat.completions.create(model=model, messages=msgs, temperature=temperature, **kw)
        return resp.choices[0].message.content
    raise ValueError(f"unsupported LLM provider {provider}")


_claude_client = None
# Claude 5-family models: effort control and server-side refusal fallback
CLAUDE_5 = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")


def _claude(model, prompt, system, json_mode):
    """Claude via the Anthropic SDK (ANTHROPIC_API_KEY). Sampling parameters are not sent: Claude
    Opus 5.5 rejects them; output_config.effort controls depth instead."""
    global _claude_client
    import anthropic

    if _claude_client is None:
        _claude_client = anthropic.Anthropic(max_retries=6)  # SDK retries 429 / 529 overloaded / 5xx
    if json_mode:  # no assistant prefill on current models: instruct, then extract the object
        system = (system + "\n\n" if system else "") + "Respond with a single JSON object only: no prose, no code fences."
    kw = {"system": system} if system else {}
    if model in CLAUDE_5:
        kw["output_config"] = {"effort": config.CLAUDE_EFFORT}
        # on a safety-classifier decline, re-run on a fallback model inside the same call
        kw["betas"] = ["server-side-fallback-2026-07-01"]
        kw["fallbacks"] = "default"
    resp = _claude_client.beta.messages.create(
        model=model, max_tokens=16000, messages=[{"role": "user", "content": prompt}], **kw)
    if resp.stop_reason == "refusal":
        category = resp.stop_details.category if resp.stop_details else None
        raise RuntimeError(f"Claude declined the request (category: {category})")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _json_object(text) if json_mode else text


def _json_object(text):
    """The JSON object in a reply, tolerating code fences or surrounding prose."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


_local_model = None


def _local():
    global _local_model
    if _local_model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _local_model = SentenceTransformer(config.EMBED_MODEL, device=device)
        _local_model.max_seq_length = config.EMBED_MAX_TOKENS
    return _local_model


def embed(texts, task="document"):
    """Embedding vectors. task='query' for search queries, 'document' for indexed chunks."""
    if config.EMBED_PROVIDER == "local":
        prefix = config.EMBED_QUERY_PREFIX if task == "query" else config.EMBED_DOC_PREFIX
        vecs = _local().encode([prefix + t for t in texts], batch_size=16, normalize_embeddings=True)
        return vecs.tolist()
    if config.EMBED_PROVIDER == "gemini":
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        reqs = [{"model": f"models/{config.EMBED_MODEL}", "content": {"parts": [{"text": t}]},
                 "taskType": task_type, "outputDimensionality": config.EMBED_DIM} for t in texts]
        resp = _gemini(config.EMBED_MODEL, "batchEmbedContents", {"requests": reqs})
        return [e["values"] for e in resp["embeddings"]]
    if config.EMBED_PROVIDER == "openai":
        from openai import OpenAI

        resp = OpenAI().embeddings.create(model=config.EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    raise ValueError(f"unsupported EMBED_PROVIDER {config.EMBED_PROVIDER}")
