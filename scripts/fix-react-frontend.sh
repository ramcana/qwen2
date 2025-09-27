#!/bin/bash
# Comprehensive React Frontend Fix Script

echo "🔧 Fixing React Frontend Dependencies..."

cd frontend

echo "📦 Backing up current setup..."
cp package.json package.json.backup

echo "🧹 Cleaning old dependencies..."
rm -rf node_modules package-lock.json

echo "📝 Updating package.json for compatibility..."

# Update package.json with compatible versions
cat > package.json << 'EOF'
{
    "name": "qwen-image-react-ui",
    "version": "2.0.0",
    "description": "Modern React frontend for Qwen-Image Generator",
    "private": true,
    "dependencies": {
        "autoprefixer": "^10.4.16",
        "axios": "^1.12.2",
        "clsx": "^2.0.0",
        "lucide-react": "^0.544.0",
        "postcss": "^8.4.31",
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "react-dropzone": "^14.2.3",
        "react-hook-form": "^7.63.0",
        "react-hot-toast": "^2.6.0",
        "react-query": "^3.39.3",
        "react-scripts": "4.0.3",
        "tailwindcss": "^3.3.0"
    },
    "devDependencies": {
        "@testing-library/jest-dom": "^5.16.4",
        "@testing-library/react": "^13.4.0",
        "@testing-library/user-event": "^14.5.1",
        "@types/jest": "^27.5.2",
        "@types/node": "^16.18.0",
        "@types/react": "^18.0.28",
        "@types/react-dom": "^18.0.11",
        "typescript": "^4.9.5"
    },
    "scripts": {
        "start": "GENERATE_SOURCEMAP=false react-scripts start",
        "build": "GENERATE_SOURCEMAP=false react-scripts build",
        "test": "react-scripts test",
        "eject": "react-scripts eject",
        "format": "prettier --write \"src/**/*.{js,jsx,ts,tsx,json,css,md}\"",
        "lint": "eslint src --ext .js,.jsx,.ts,.tsx"
    },
    "browserslist": {
        "production": [
            ">0.2%",
            "not dead",
            "not op_mini all"
        ],
        "development": [
            "last 1 chrome version",
            "last 1 firefox version",
            "last 1 safari version"
        ]
    },
    "resolutions": {
        "schema-utils": "3.3.0"
    }
}
EOF

echo "📥 Installing compatible dependencies..."
npm install --legacy-peer-deps --no-audit --no-fund

echo "🔧 Creating environment file..."
cat > .env << 'EOF'
GENERATE_SOURCEMAP=false
SKIP_PREFLIGHT_CHECK=true
REACT_APP_API_BASE_URL=http://localhost:8000
EOF

echo "✅ Frontend fixed! Starting development server..."
echo "🌐 Frontend will be available at: http://localhost:3000"
echo ""

npm start