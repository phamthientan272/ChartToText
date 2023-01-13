from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import csv
import traceback
from io import BytesIO
import pandas as pd
from preprocessing.preprocess import Preprocessor
from model.summarize import Model
from postprocessing.postprocess import post_processing
import nltk

router = APIRouter()
model = Model("model/aug17-80.pth")
nltk.download('punkt')
@router.post("/describe/")
async def describe(title: str = Form(...), file: UploadFile = File(...)):
    try:
        
        contents = file.file.read()
        raw_data_file = BytesIO(contents)
        
        preprocessor = Preprocessor(raw_data_file, title)
        data, title = preprocessor.preprocess_data()
        caption = model.describe(data, title)
        output = post_processing(caption, data, title)
        
        raw_data_file.close()
        file.file.close()
        
        return JSONResponse({"title": title, "data": data, "caption": output}) 
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
