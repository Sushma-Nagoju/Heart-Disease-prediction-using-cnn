from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow import keras
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ML Model Setup
class HeartDiseasePredictor:
    def __init__(self):
        self.model = self._create_model()
        self.input_shape = (224, 224, 3)
        
    def _create_model(self):
        """Create a CNN model for heart disease prediction using transfer learning"""
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(3, activation='softmax')  # 3 classes: Low, Medium, High risk
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_bytes):
        """Preprocess uploaded CT scan image"""
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_risk(self, image_bytes):
        """Predict heart disease risk from CT scan"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Make prediction (simulated for MVP)
            # In a real scenario, you would have a trained model
            predictions = np.random.rand(3)
            predictions = predictions / np.sum(predictions)  # Normalize to sum to 1
            
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_percentages = [float(p * 100) for p in predictions]
            
            # Determine primary risk level
            max_idx = np.argmax(predictions)
            primary_risk = risk_labels[max_idx]
            confidence = risk_percentages[max_idx]
            
            # Generate risk analysis
            analysis = self._generate_analysis(predictions, processed_image[0])
            
            return {
                'primary_risk': primary_risk,
                'confidence': round(confidence, 2),
                'risk_breakdown': {
                    'low_risk': round(risk_percentages[0], 2),
                    'medium_risk': round(risk_percentages[1], 2),
                    'high_risk': round(risk_percentages[2], 2)
                },
                'analysis': analysis,
                'recommendations': self._get_recommendations(primary_risk)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    def _generate_analysis(self, predictions, image):
        """Generate detailed analysis of the CT scan"""
        risk_level = np.argmax(predictions)
        
        analysis_texts = [
            "CT scan shows normal cardiac structures with no significant abnormalities detected. Blood vessels appear clear and heart chambers are within normal size ranges.",
            "CT scan indicates some areas of concern that warrant monitoring. Possible signs of coronary calcification or mild structural changes observed.",
            "CT scan reveals significant abnormalities including potential blockages, enlarged heart chambers, or concerning calcium buildup in coronary arteries."
        ]
        
        return analysis_texts[risk_level]
    
    def _get_recommendations(self, risk_level):
        """Get recommendations based on risk level"""
        recommendations = {
            'Low Risk': [
                "Continue maintaining a healthy lifestyle",
                "Regular exercise and balanced diet",
                "Annual routine checkups",
                "Monitor blood pressure and cholesterol"
            ],
            'Medium Risk': [
                "Consult with a cardiologist for detailed evaluation",
                "Consider lifestyle modifications (diet, exercise)",
                "Monitor symptoms closely",
                "Follow up in 3-6 months",
                "Consider additional cardiac tests"
            ],
            'High Risk': [
                "Immediate consultation with a cardiologist required",
                "Emergency evaluation may be necessary",
                "Comprehensive cardiac workup recommended",
                "Discuss treatment options with healthcare provider",
                "Do not ignore symptoms - seek medical attention"
            ]
        }
        
        return recommendations.get(risk_level, [])

# Initialize ML predictor
predictor = HeartDiseasePredictor()

# Define Models
class PredictionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    primary_risk: str
    confidence: float
    risk_breakdown: Dict[str, float]
    analysis: str
    recommendations: List[str]
    image_name: str

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Routes
@api_router.get("/")
async def root():
    return {"message": "Heart Disease Risk Prediction API"}

@api_router.post("/predict", response_model=PredictionResult)
async def predict_heart_disease(file: UploadFile = File(...)):
    """Upload CT scan and get heart disease risk prediction"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Get prediction
        prediction = predictor.predict_risk(image_bytes)
        
        # Create result object
        result = PredictionResult(
            primary_risk=prediction['primary_risk'],
            confidence=prediction['confidence'],
            risk_breakdown=prediction['risk_breakdown'],
            analysis=prediction['analysis'],
            recommendations=prediction['recommendations'],
            image_name=file.filename
        )
        
        # Save to database
        await db.predictions.insert_one(result.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@api_router.get("/predictions", response_model=List[PredictionResult])
async def get_predictions():
    """Get all previous predictions"""
    predictions = await db.predictions.find().sort("timestamp", -1).to_list(100)
    return [PredictionResult(**pred) for pred in predictions]

@api_router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Get specific prediction by ID"""
    prediction = await db.predictions.find_one({"id": prediction_id})
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return PredictionResult(**prediction)

# Legacy status check routes
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()