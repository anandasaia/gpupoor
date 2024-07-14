from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
import uuid
import json
from fastapi.responses import FileResponse, StreamingResponse
import os
from zipfile import ZipFile
import subprocess
import threading
import queue
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse

TRAINING_RUNS_FILE = "training_runs.json"

app = FastAPI()

# Allow all origins - you might want to restrict this to specific domains in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as necessary for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Path to the JSON database
db_path = "model_data.json"

# Ensure the database file exists
if not os.path.exists(db_path):
    with open(db_path, 'w') as db_file:
        json.dump({}, db_file)

def read_db():
    with open(db_path, 'r') as db_file:
        return json.load(db_file)

def write_db(data):
    with open(db_path, 'w') as db_file:
        json.dump(data, db_file, indent=4)


def read_training_runs_db():
    if not os.path.exists(TRAINING_RUNS_FILE):
        return {}
    with open(TRAINING_RUNS_FILE, "r") as file:
        return json.load(file)

def write_training_runs_db(db):
    with open(TRAINING_RUNS_FILE, "w") as file:
        json.dump(db, file)

class ModelFund(BaseModel):
    eth_amount: float
    model_file: str

class ClientRequest(BaseModel):
    model_id: str
    percentage: int
    eth_address: str

@app.post("/fund-model/")
async def fund_model(model_fund: ModelFund):
    db = read_db()
    model_id = str(uuid.uuid4())
    db[model_id] = {
        "eth_amount": model_fund.eth_amount,
        "model_file": model_fund.model_file,
        "total_allocated": 0,
        "clients": {}
    }
    write_db(db)
    return {"model_id": model_id}

def generate_client_file(model_id, percentage, eth_address):
    # Generate the client.py file based on the inputs (mock implementation)
    client_file_path = f"client_{model_id}_{eth_address}.py"
    with open(client_file_path, "w") as file:
        file.write(f"# Client file for model {model_id} with {percentage}% allocation for {eth_address}\n")
    return client_file_path

class ClientRequest(BaseModel):
    model_id: str
    percentage: int
    eth_address: str

@app.post("/generate-client/")
async def generate_client(client_request: ClientRequest):
    db = read_training_runs_db()
    if client_request.model_id not in db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = db[client_request.model_id]
    if model["total_allocated"] + client_request.percentage > 100:
        raise HTTPException(status_code=400, detail="Requested percentage exceeds available dataset")
    
    model["total_allocated"] += client_request.percentage
    model["clients"][client_request.eth_address] = client_request.percentage
    
    write_training_runs_db(db)
    
    # Generate client file
    client_file = generate_client_file(client_request.model_id, client_request.percentage, client_request.eth_address)
    
    # Create a zip file containing the client.py, model.py, requirements.txt, and README.md
    zip_filename = f"client_package_{client_request.model_id}_{client_request.eth_address}.zip"
    with ZipFile(zip_filename, 'w') as zipf:
        zipf.write(client_file, arcname="client.py")
        zipf.write("model.py", arcname="model.py")
        zipf.write("requirements.txt", arcname="requirements.txt")
        zipf.write("README.md", arcname="README.md")
        zipf.write("install_and_run.sh", arcname="install_and_run.sh")
    
    return FileResponse(zip_filename, filename=zip_filename)

@app.get("/training-runs")
async def get_training_runs():
    db = read_training_runs_db()
    return JSONResponse(content=db, status_code=200)

@app.post("/training-runs")
async def add_training_run(run: dict):
    db = read_training_runs_db()
    db.append(run)
    write_training_runs_db(db)
    return JSONResponse(content={"message": "Training run added successfully"}, status_code=201)

@app.delete("/training-runs/{run_id}")
async def delete_training_run(run_id: int):
    db = read_training_runs_db()
    db = [run for run in db if run["id"] != run_id]
    write_training_runs_db(db)
    return JSONResponse(content={"message": "Training run deleted successfully"}, status_code=200)


@app.get("/model-status/{model_id}")
async def model_status(model_id: str):
    db = read_db()
    if model_id not in db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = db[model_id]
    if model["total_allocated"] >= 100:
        return {"status": "Model run filled"}
    else:
        return {"status": "Available", "percentage_left": 100 - model["total_allocated"]}


def run_server():
    """Function to run the server and capture its output."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    server_script_path = os.path.join(dir_path, "server.py")
    cmd = ["python", server_script_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in proc.stdout:
        yield f"data: {line}\n\n"  # Format the output as server-sent events
    proc.stdout.close()
    proc.wait()

@app.get("/start-server/")
async def start_server():
    """Endpoint to start the server and stream logs."""
    return StreamingResponse(run_server(), media_type="text/event-stream")




        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)