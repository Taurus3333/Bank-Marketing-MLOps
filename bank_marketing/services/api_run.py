# bank_marketing/services/api_run.py
from __future__ import annotations
import uvicorn

def main():
    uvicorn.run("bank_marketing.services.api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
