import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tripletex-agent")

app = FastAPI(title="Tripletex AI Agent")


@app.post("/solve")
async def solve(request: Request):
    try:
        body = await request.json()
        prompt = body["prompt"]
        files = body.get("files", [])
        creds = body["tripletex_credentials"]

        base_url = creds["base_url"]
        session_token = creds["session_token"]

        logger.info("Received task: %s", prompt[:120])
        logger.info("Credentials: base_url=%s token=%s...", base_url, session_token[:8] if session_token else "NONE")

        await run_agent(
            prompt=prompt,
            files=files,
            base_url=base_url,
            session_token=session_token,
        )

        logger.info("Task completed successfully")
    except Exception:
        logger.error("Agent failed:\n%s", traceback.format_exc())

    return JSONResponse({"status": "completed"})
