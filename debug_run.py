import uvicorn
from main import app
import logging

if __name__ == "__main__":
    # Just run the server directly
    uvicorn.run(app, host="127.0.0.1", port=8000)
