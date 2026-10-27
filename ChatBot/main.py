import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run(
        "ADKBot.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 6655)),
        reload=True
    )