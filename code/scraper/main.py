from fastapi import FastAPI
import httpx
app = FastAPI()
import asyncio
from contextlib import asynccontextmanager

VBB_URL = "https://vbb-demo.demo2.hafas.cloud/api/fahrinfo/latest/"
ACCESS_ID = "wolf-4af9-884e-91a1fe245e62"
fetch_task = None 
latest_data = None

async def fetch_loop ():
    global latest_data
    async with httpx.AsyncClient (timeout= 10) as client: 
        while True: 
          try: 
            result = await client.get(VBB_URL, params={"accessId":ACCESS_ID})
            result.raise_for_status()
            latest_data = {
                "status": "success",
                "status_code": result.status_code,
                "bytes": len(result.content),
                "data": result.json()
            }
          except httpx.HTTPStatusError as error: 
              latest_data = {"status": "status error", "code": error.response.status_code}
          except Exception as error:
           latest_data ={"status": "error", "message": str(error)}
          await asyncio.sleep(1)

@asynccontextmanager
async def lifespan (app:FastAPI):
   global fetch_task #Aenderung der globalen Variable
   fetch_task = asyncio.create_task(fetch_loop())
   print("Fetch läuft: 60 reqs/Minute")
   yield 
   fetch_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/fetch")
async def status ():
   return latest_data or {"status": "waiting"}