#!/usr/bin/env node

/**
 * Container Communication Configuration Validator
 * 
 * This script validates the frontend-backend container communication configuration
 * by checking configuration files, environment variables, and Docker setup.
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating Container Communication Configuration');
console.log('================================================');

let validationErrors = [];
let validationWarnings = [];

// Check if file exists
function checkFile(filePath, description) {
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${description}: ${filePath}`);
    return true;
  } else {
    console.log(`❌ ${description}: ${filePath} (NOT FOUND)`);
    validationErrors.push(`Missing file: ${filePath}`);
    return false;
  }
}

// Check file content for specific patterns
function checkFileContent(filePath, patterns, description) {
  if (!fs.existsSync(filePath)) {
    return false;
  }
  
  const content = fs.readFileSync(filePath, 'utf8');
  const missingPatterns = [];
  
  for (const [pattern, desc] of patterns) {
    if (!content.includes(pattern)) {
      missingPatterns.push(desc);
    }
  }
  
  if (missingPatterns.length === 0) {
    console.log(`✅ ${description}: All required patterns found`);
    return true;
  } else {
    console.log(`⚠️  ${description}: Missing patterns - ${missingPatterns.join(', ')}`);
    validationWarnings.push(`${description}: Missing - ${missingPatterns.join(', ')}`);
    return false;
  }
}

console.log('\n📁 File Existence Checks');
console.log('------------------------');

// Check required configuration files
checkFile('nginx.prod.conf', 'Nginx production configuration');
checkFile('nginx.conf', 'Nginx development configuration');
checkFile('api_error.html', 'API error page');
checkFile('src/config/api.ts', 'API configuration service');
checkFile('src/services/api.ts', 'API service');
checkFile('.env', 'Frontend environment variables');
checkFile('../docker-compose.yml', 'Main Docker Compose file');
checkFile('CONTAINER_COMMUNICATION.md', 'Documentation');

console.log('\n🔧 Configuration Content Checks');
console.log('-------------------------------');

// Check nginx configuration
checkFileContent('nginx.prod.conf', [
  ['proxy_pass http://qwen-api:8000/', 'API proxy configuration'],
  ['Access-Control-Allow-Origin', 'CORS headers'],
  ['location /api/', 'API location block'],
  ['location /ws', 'WebSocket location block'],
  ['$cors_origin', 'CORS origin mapping']
], 'Nginx production config');

// Check API configuration
checkFileContent('src/config/api.ts', [
  ['apiConfig', 'API configuration export'],
  ['REACT_APP_API_URL', 'API URL environment variable'],
  ['REACT_APP_BACKEND_HOST', 'Backend host environment variable'],
  ['getImageUrl', 'Image URL helper function']
], 'API configuration service');

// Check API service
checkFileContent('src/services/api.ts', [
  ['withCredentials: true', 'CORS credentials'],
  ['apiConfig', 'Configuration import'],
  ['axios.create', 'Axios instance creation']
], 'API service');

// Check environment variables
checkFileContent('.env', [
  ['REACT_APP_API_URL', 'API URL variable'],
  ['REACT_APP_BACKEND_HOST', 'Backend host variable'],
  ['qwen-api', 'Backend service name']
], 'Frontend environment variables');

// Check Docker Compose
checkFileContent('../docker-compose.yml', [
  ['REACT_APP_CONTAINER_MODE=true', 'Container mode flag'],
  ['REACT_APP_API_URL=/api', 'Proxied API URL'],
  ['REACT_APP_BACKEND_HOST=qwen-api', 'Backend host configuration'],
  ['qwen-network', 'Docker network configuration']
], 'Docker Compose configuration');

console.log('\n🐳 Docker Configuration Checks');
console.log('------------------------------');

// Check Docker network configuration
if (fs.existsSync('../docker-compose.yml')) {
  const dockerContent = fs.readFileSync('../docker-compose.yml', 'utf8');
  
  // Check for network configuration
  if (dockerContent.includes('qwen-network')) {
    console.log('✅ Docker network: qwen-network configured');
  } else {
    console.log('❌ Docker network: qwen-network not found');
    validationErrors.push('Missing Docker network configuration');
  }
  
  // Check for service dependencies
  if (dockerContent.includes('depends_on:') && dockerContent.includes('api:')) {
    console.log('✅ Service dependencies: Frontend depends on API');
  } else {
    console.log('⚠️  Service dependencies: Frontend-API dependency not explicit');
    validationWarnings.push('Missing explicit service dependencies');
  }
}

console.log('\n📊 Validation Summary');
console.log('====================');

console.log(`Total Errors: ${validationErrors.length}`);
console.log(`Total Warnings: ${validationWarnings.length}`);

if (validationErrors.length > 0) {
  console.log('\n❌ Validation Errors:');
  validationErrors.forEach((error, index) => {
    console.log(`   ${index + 1}. ${error}`);
  });
}

if (validationWarnings.length > 0) {
  console.log('\n⚠️  Validation Warnings:');
  validationWarnings.forEach((warning, index) => {
    console.log(`   ${index + 1}. ${warning}`);
  });
}

if (validationErrors.length === 0) {
  console.log('\n🎉 Configuration validation passed!');
  console.log('   Frontend-backend container communication should work correctly.');
  
  if (validationWarnings.length > 0) {
    console.log('   Note: There are some warnings that should be addressed for optimal performance.');
  }
  
  console.log('\n📋 Next Steps:');
  console.log('   1. Build the frontend container: docker-compose build frontend');
  console.log('   2. Start the services: docker-compose up -d');
  console.log('   3. Test communication: node frontend/test-container-communication.js');
  
  process.exit(0);
} else {
  console.log('\n💥 Configuration validation failed!');
  console.log('   Please fix the errors above before proceeding.');
  process.exit(1);
}