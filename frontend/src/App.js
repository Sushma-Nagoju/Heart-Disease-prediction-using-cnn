import React, { useState, useEffect } from 'react';
import './App.css';
import { Upload, Heart, Activity, AlertTriangle, CheckCircle, FileImage, Camera, TrendingUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Alert, AlertDescription } from './components/ui/alert';
import { Progress } from './components/ui/progress';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Separator } from './components/ui/separator';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState(null);

  // Fetch previous predictions on load
  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const response = await axios.get(`${API}/predictions`);
      setPredictions(response.data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!file) {
      setError('Please select a CT scan image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
      await fetchPredictions(); // Refresh predictions list
    } catch (error) {
      console.error('Error:', error);
      setError(error.response?.data?.detail || 'Error processing image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'Low Risk':
        return 'text-green-600 bg-green-50';
      case 'Medium Risk':
        return 'text-yellow-600 bg-yellow-50';
      case 'High Risk':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'Low Risk':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'Medium Risk':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'High Risk':
        return <AlertTriangle className="w-5 h-5 text-red-600" />;
      default:
        return <Heart className="w-5 h-5" />;
    }
  };

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">CardioScan AI</h1>
                <p className="text-sm text-gray-600">Heart Disease Risk Prediction</p>
              </div>
            </div>
            <Badge variant="secondary" className="flex items-center gap-2">
              <Activity className="w-4 h-4" />
              AI-Powered Analysis
            </Badge>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="analyze" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="analyze" className="flex items-center gap-2">
              <Camera className="w-4 h-4" />
              New Analysis
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="analyze" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Upload Section */}
              <Card className="h-fit">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Upload CT Scan
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                    {preview ? (
                      <div className="space-y-4">
                        <img 
                          src={preview} 
                          alt="CT Scan Preview" 
                          className="max-w-full h-48 mx-auto object-contain rounded-lg shadow-sm"
                        />
                        <p className="text-sm text-gray-600">{file?.name}</p>
                        <Button 
                          variant="outline" 
                          onClick={resetForm}
                          className="w-full"
                        >
                          Choose Different Image
                        </Button>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <FileImage className="w-12 h-12 text-gray-400 mx-auto" />
                        <div>
                          <p className="text-lg font-medium text-gray-900">
                            Select CT Scan Image
                          </p>
                          <p className="text-sm text-gray-600">
                            Upload a clear CT scan image for analysis
                          </p>
                        </div>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleFileSelect}
                          className="hidden"
                          id="file-upload"
                        />
                        <label
                          htmlFor="file-upload"
                          className="cursor-pointer inline-flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Choose File
                        </label>
                      </div>
                    )}
                  </div>

                  {error && (
                    <Alert variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}

                  <Button
                    onClick={handleSubmit}
                    disabled={!file || loading}
                    className="w-full bg-blue-600 hover:bg-blue-700"
                  >
                    {loading ? (
                      <div className="flex items-center gap-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                        Analyzing...
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        Analyze CT Scan
                      </div>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Results Section */}
              {result && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {getRiskIcon(result.primary_risk)}
                      Analysis Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Primary Risk */}
                    <div className="text-center">
                      <Badge className={`text-lg px-4 py-2 ${getRiskColor(result.primary_risk)}`}>
                        {result.primary_risk}
                      </Badge>
                      <p className="text-2xl font-bold mt-2">{result.confidence}% Confidence</p>
                    </div>

                    <Separator />

                    {/* Risk Breakdown */}
                    <div className="space-y-3">
                      <h4 className="font-semibold">Risk Breakdown</h4>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-green-600">Low Risk</span>
                          <span>{result.risk_breakdown.low_risk}%</span>
                        </div>
                        <Progress value={result.risk_breakdown.low_risk} className="h-2" />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-yellow-600">Medium Risk</span>
                          <span>{result.risk_breakdown.medium_risk}%</span>
                        </div>
                        <Progress value={result.risk_breakdown.medium_risk} className="h-2" />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-red-600">High Risk</span>
                          <span>{result.risk_breakdown.high_risk}%</span>
                        </div>
                        <Progress value={result.risk_breakdown.high_risk} className="h-2" />
                      </div>
                    </div>

                    <Separator />

                    {/* Analysis */}
                    <div>
                      <h4 className="font-semibold mb-2">Detailed Analysis</h4>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {result.analysis}
                      </p>
                    </div>

                    {/* Recommendations */}
                    <div>
                      <h4 className="font-semibold mb-2">Recommendations</h4>
                      <ul className="space-y-1">
                        {result.recommendations.map((rec, index) => (
                          <li key={index} className="text-sm text-gray-700 flex items-start gap-2">
                            <span className="text-blue-600 mt-1">•</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="history">
            <Card>
              <CardHeader>
                <CardTitle>Previous Analyses</CardTitle>
              </CardHeader>
              <CardContent>
                {predictions.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No previous analyses found</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {predictions.map((pred) => (
                      <div key={pred.id} className="border rounded-lg p-4 hover:bg-gray-50">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {getRiskIcon(pred.primary_risk)}
                            <div>
                              <Badge className={`${getRiskColor(pred.primary_risk)}`}>
                                {pred.primary_risk}
                              </Badge>
                              <p className="text-sm text-gray-600 mt-1">
                                {pred.image_name} • {pred.confidence}% confidence
                              </p>
                            </div>
                          </div>
                          <div className="text-sm text-gray-500">
                            {new Date(pred.timestamp).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;