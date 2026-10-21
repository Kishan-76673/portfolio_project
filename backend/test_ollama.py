import asyncio
import time
import httpx

async def test_ollama():
    print("Testing Ollama connectivity on http://localhost:11434...")
    t0 = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:1b",
                "messages": [{"role": "user", "content": "Respond with 'Ollama is alive!'"}],
                "stream": False,
            }
        )
        elapsed = time.time() - t0
        print(f"Status Code: {res.status_code}")
        print(f"Response Time: {elapsed:.2f}s")
        if res.status_code == 200:
            print("Content:", res.json().get("message", {}).get("content"))
        else:
            print("Error:", res.text)

if __name__ == "__main__":
    asyncio.run(test_ollama())
