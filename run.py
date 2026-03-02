import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the server
    # host="0.0.0.0" makes the server accessible from other devices on the network
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
