from fastapi import FastAPI
from app.routers import function
from app.utils import auth, logger
from app.functions import user_functions  

app = FastAPI()
app.include_router(function.router)


logger.setup_logging()
auth.init_security_keys()

@app.get("/health")
def health_check():
    return {"status": "ok"}