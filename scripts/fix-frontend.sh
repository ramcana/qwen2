#!/bin/bash
# Fix React frontend dependencies

echo "🔧 Fixing React frontend dependencies..."

cd frontend

echo "📦 Removing old dependencies..."
rm -rf node_modules package-lock.json

echo "📥 Installing dependencies with legacy peer deps..."
npm install --legacy-peer-deps

echo "✅ Frontend dependencies fixed!"
echo "🚀 Starting React development server..."
npm start