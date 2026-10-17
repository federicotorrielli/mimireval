"""Client for the OpenAI-compatible judge API, with an on-disk cache.

Batch calls run async (`judge(prompts)`); single calls are sync and thread-safe
(`judge.one(prompt)`), for third-party code that uses thread pools.
"""
import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from mimireval.models import PRIMARY_JUDGE, SECOND_JUDGE  # noqa: F401 (re-exported)
CACHE_DIR = Path(os.environ.get("PROJ", ".")) / "cache" / "judge"
_retry = retry(stop=stop_after_attempt(20), wait=wait_exponential(min=2, max=300), reraise=True)  # the shared API stalls at peak


class Judge:
    def __init__(self, model: str = PRIMARY_JUDGE, concurrency: int = 64, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.lock = threading.Lock()
        self.sync_client = openai.OpenAI(base_url=os.environ["JUDGE_BASE_URL"],
                                         api_key=os.environ["JUDGE_API_KEY"], timeout=180)  # a reply takes <~80 s; fail fast on a hung request
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"{model.replace('/', '__')}.jsonl"
        self.usage_path = self.cache_path.with_suffix(".usage.jsonl")
        self.cache = {}
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # last line cut by a killed job
                        continue
                    self.cache[rec["key"]] = rec["content"]

    def _key(self, prompt: str, temperature: float, top_p: float) -> str:
        return hashlib.sha256(f"{self.model}\x00{temperature}\x00{top_p}\x00{prompt}".encode()).hexdigest()

    def _store(self, key: str, content: str) -> str:
        with self.lock:
            self.cache[key] = content
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"key": key, "content": content}) + "\n")
        return content

    def _request(self, prompt, temperature, top_p):
        return dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, top_p=top_p, max_tokens=self.max_tokens)

    def _content(self, r) -> str:
        if r.usage:  # billing record: reasoning tokens are part of completion_tokens
            with self.lock, self.usage_path.open("a") as f:
                f.write(json.dumps({"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}) + "\n")
        content = r.choices[0].message.content
        if content is None:
            raise RuntimeError("empty judge content (reasoning hit max_tokens?)")
        return content

    # sync, thread-safe
    @_retry
    def _call_sync(self, prompt, temperature, top_p) -> str:
        return self._content(self.sync_client.chat.completions.create(**self._request(prompt, temperature, top_p)))

    def one(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> str:
        key = self._key(prompt, temperature, top_p)
        if key in self.cache:
            return self.cache[key]
        return self._store(key, self._call_sync(prompt, temperature, top_p))

    # async batch
    @_retry
    async def _call_async(self, prompt, temperature, top_p) -> str:
        async with self.sem:
            return self._content(await self.client.chat.completions.create(**self._request(prompt, temperature, top_p)))

    async def _one_async(self, prompt, temperature, top_p) -> str:
        key = self._key(prompt, temperature, top_p)
        if key in self.cache:
            return self.cache[key]
        return self._store(key, await self._call_async(prompt, temperature, top_p))

    async def _many(self, prompts, temperature, top_p, desc) -> list[str]:
        # Loop-bound objects are created per asyncio.run.
        self.sem = asyncio.Semaphore(self.concurrency)
        self.client = openai.AsyncOpenAI(base_url=os.environ["JUDGE_BASE_URL"],
                                         api_key=os.environ["JUDGE_API_KEY"], timeout=180)  # a reply takes <~80 s; fail fast on a hung request
        try:
            return await tqdm_asyncio.gather(*(self._one_async(p, temperature, top_p) for p in prompts), desc=desc)
        finally:
            await self.client.close()  # before asyncio.run closes the loop

    def __call__(self, prompts: list[str], temperature: float = 0.0, top_p: float = 1.0, desc: str = "judge") -> list[str]:
        return asyncio.run(self._many(prompts, temperature, top_p, f"{desc} [{self.model}]"))
