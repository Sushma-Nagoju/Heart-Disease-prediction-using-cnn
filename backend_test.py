import requests
import sys
import os
from datetime import datetime
import json
from PIL import Image
import io
import base64

class HeartDiseaseAPITester:
    def __init__(self, base_url="https://0bc2714d-6ec0-40e2-9a4c-6688c9b9f0f4.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.prediction_id = None

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def create_test_image(self):
        """Create a simple test image for upload"""
        # Create a simple 224x224 RGB image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/")
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Message: {data.get('message', 'No message')}"
            return self.log_test("Root Endpoint", success, details)
        except Exception as e:
            return self.log_test("Root Endpoint", False, f"Error: {str(e)}")

    def test_predict_endpoint(self):
        """Test the prediction endpoint with image upload"""
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare multipart form data
            files = {'file': ('test_ct_scan.jpg', test_image, 'image/jpeg')}
            
            response = requests.post(f"{self.api_url}/predict", files=files)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                # Store prediction ID for later tests
                self.prediction_id = data.get('id')
                
                # Validate response structure
                required_fields = ['id', 'primary_risk', 'confidence', 'risk_breakdown', 'analysis', 'recommendations', 'image_name']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    success = False
                    details = f"Missing fields: {missing_fields}"
                else:
                    # Validate risk breakdown structure
                    risk_breakdown = data.get('risk_breakdown', {})
                    risk_fields = ['low_risk', 'medium_risk', 'high_risk']
                    missing_risk_fields = [field for field in risk_fields if field not in risk_breakdown]
                    
                    if missing_risk_fields:
                        success = False
                        details = f"Missing risk breakdown fields: {missing_risk_fields}"
                    else:
                        details = f"Risk: {data['primary_risk']}, Confidence: {data['confidence']}%"
                        
            else:
                details = f"Status: {response.status_code}, Error: {response.text}"
                
            return self.log_test("Predict Endpoint", success, details)
            
        except Exception as e:
            return self.log_test("Predict Endpoint", False, f"Error: {str(e)}")

    def test_get_predictions(self):
        """Test getting all predictions"""
        try:
            response = requests.get(f"{self.api_url}/predictions")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Found {len(data)} predictions"
                
                # Validate that our prediction is in the list
                if self.prediction_id:
                    found_prediction = any(pred.get('id') == self.prediction_id for pred in data)
                    if not found_prediction:
                        success = False
                        details += ", but our test prediction was not found"
                    else:
                        details += ", including our test prediction"
            else:
                details = f"Status: {response.status_code}, Error: {response.text}"
                
            return self.log_test("Get All Predictions", success, details)
            
        except Exception as e:
            return self.log_test("Get All Predictions", False, f"Error: {str(e)}")

    def test_get_specific_prediction(self):
        """Test getting a specific prediction by ID"""
        if not self.prediction_id:
            return self.log_test("Get Specific Prediction", False, "No prediction ID available from previous test")
            
        try:
            response = requests.get(f"{self.api_url}/predictions/{self.prediction_id}")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Retrieved prediction: {data.get('primary_risk', 'Unknown')} risk"
                
                # Validate that the ID matches
                if data.get('id') != self.prediction_id:
                    success = False
                    details = f"ID mismatch: expected {self.prediction_id}, got {data.get('id')}"
            else:
                details = f"Status: {response.status_code}, Error: {response.text}"
                
            return self.log_test("Get Specific Prediction", success, details)
            
        except Exception as e:
            return self.log_test("Get Specific Prediction", False, f"Error: {str(e)}")

    def test_invalid_file_upload(self):
        """Test uploading non-image file"""
        try:
            # Create a text file instead of image
            text_content = b"This is not an image file"
            files = {'file': ('test.txt', text_content, 'text/plain')}
            
            response = requests.post(f"{self.api_url}/predict", files=files)
            success = response.status_code == 400  # Should return 400 for invalid file type
            
            details = f"Status: {response.status_code}"
            if success:
                details += " (correctly rejected non-image file)"
            else:
                details += " (should have rejected non-image file)"
                
            return self.log_test("Invalid File Upload", success, details)
            
        except Exception as e:
            return self.log_test("Invalid File Upload", False, f"Error: {str(e)}")

    def test_missing_file_upload(self):
        """Test prediction endpoint without file"""
        try:
            response = requests.post(f"{self.api_url}/predict")
            success = response.status_code == 422  # Should return 422 for missing required field
            
            details = f"Status: {response.status_code}"
            if success:
                details += " (correctly rejected missing file)"
            else:
                details += " (should have rejected missing file)"
                
            return self.log_test("Missing File Upload", success, details)
            
        except Exception as e:
            return self.log_test("Missing File Upload", False, f"Error: {str(e)}")

    def test_nonexistent_prediction(self):
        """Test getting a non-existent prediction"""
        try:
            fake_id = "nonexistent-prediction-id"
            response = requests.get(f"{self.api_url}/predictions/{fake_id}")
            success = response.status_code == 404  # Should return 404 for not found
            
            details = f"Status: {response.status_code}"
            if success:
                details += " (correctly returned 404 for non-existent prediction)"
            else:
                details += " (should have returned 404 for non-existent prediction)"
                
            return self.log_test("Non-existent Prediction", success, details)
            
        except Exception as e:
            return self.log_test("Non-existent Prediction", False, f"Error: {str(e)}")

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🔬 Starting Heart Disease API Backend Tests")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Core functionality tests
        self.test_root_endpoint()
        self.test_predict_endpoint()
        self.test_get_predictions()
        self.test_get_specific_prediction()
        
        # Error handling tests
        self.test_invalid_file_upload()
        self.test_missing_file_upload()
        self.test_nonexistent_prediction()
        
        # Print summary
        print("=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All backend tests passed!")
            return True
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed")
            return False

def main():
    tester = HeartDiseaseAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())