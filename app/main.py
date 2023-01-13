from fastapi import Depends, FastAPI, Header
from fastapi.responses import JSONResponse

from routers import describe_chart
import uvicorn

app = FastAPI()
app.include_router(describe_chart.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5001, reload=True)

