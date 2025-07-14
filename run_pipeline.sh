#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting Housing Price Prediction Pipeline..."

# 1. Train model
echo "🔧 Step 1: Training the model..."
python scripts/train_model.py

# 2. Predict on test data
echo "📈 Step 2: Running prediction on test data..."
python scripts/predict.py

echo "✅ Pipeline completed successfully!"
