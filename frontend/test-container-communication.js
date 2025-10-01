#!/usr/bin/env node

/**
 * Container Communication Test Script
 * 
 * This script tests the frontend-backend container communication
 * by making HTTP requests to various endpoints through the nginx proxy.
 */

const http = require('http');
const https = require('https');

const TEST_CONFIG = {
  frontend: {
    host: 'localhost',
    port: 80,
    protocol: 'http'
  },
  api: {
    host: 'qwen-api',
    port: 8000,
    protocol: 'http'
  }
};

// Test endpoints
const TEST_ENDPOINTS = [
  { path: '/health', description: 'Frontend health check' },
  { path: '/api/health', description: 'API health check through proxy' },
  { path: '/api/status', description: 'API status through proxy' },
];

function makeRequest(options) {
  return new Promise((resolve, reject) => {
    const client = options.protocol === 'https' ? https : http;
    
    const req = client.request({
      hostname: options.host,
      port: options.port,
      path: options.path,
      method: 'GET',
      timeout: 10000,
      headers: {
        'User-Agent': 'Container-Communication-Test/1.0',
        'Accept': 'application/json, text/html, */*',
        'Cache-Control': 'no-cache'
      }
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers,
          data: data,
          success: res.statusCode >= 200 && res.statusCode < 400
        });
      });
    });

    req.on('error', (error) => {
      reject({
        error: error.message,
        success: false
      });
    });

    req.on('timeout', () => {
      req.destroy();
      reject({
        error: 'Request timeout',
        success: false
      });
    });

    req.end();
  });
}

async function testEndpoint(endpoint, config) {
  console.log(`\n🧪 Testing: ${endpoint.description}`);
  console.log(`   URL: ${config.protocol}://${config.host}:${config.port}${endpoint.path}`);
  
  try {
    const result = await makeRequest({
      ...config,
      path: endpoint.path
    });
    
    if (result.success) {
      console.log(`   ✅ SUCCESS (${result.statusCode})`);
      
      // Check for CORS headers
      if (result.headers['access-control-allow-origin']) {
        console.log(`   🔒 CORS: ${result.headers['access-control-allow-origin']}`);
      }
      
      // Show response preview
      if (result.data.length > 0) {
        const preview = result.data.substring(0, 100);
        console.log(`   📄 Response: ${preview}${result.data.length > 100 ? '...' : ''}`);
      }
    } else {
      console.log(`   ❌ FAILED (${result.statusCode})`);
    }
    
    return result.success;
  } catch (error) {
    console.log(`   ❌ ERROR: ${error.error || error.message}`);
    return false;
  }
}

async function runTests() {
  console.log('🚀 Container Communication Test Suite');
  console.log('=====================================');
  
  let totalTests = 0;
  let passedTests = 0;
  
  // Test frontend endpoints
  console.log('\n📱 Frontend Container Tests');
  console.log('---------------------------');
  
  for (const endpoint of TEST_ENDPOINTS.filter(e => !e.path.startsWith('/api'))) {
    totalTests++;
    const success = await testEndpoint(endpoint, TEST_CONFIG.frontend);
    if (success) passedTests++;
  }
  
  // Test API endpoints through proxy
  console.log('\n🔗 API Proxy Tests');
  console.log('------------------');
  
  for (const endpoint of TEST_ENDPOINTS.filter(e => e.path.startsWith('/api'))) {
    totalTests++;
    const success = await testEndpoint(endpoint, TEST_CONFIG.frontend);
    if (success) passedTests++;
  }
  
  // Test direct API connection (if accessible)
  console.log('\n🎯 Direct API Tests');
  console.log('-------------------');
  
  try {
    totalTests++;
    const success = await testEndpoint(
      { path: '/health', description: 'Direct API health check' },
      TEST_CONFIG.api
    );
    if (success) passedTests++;
  } catch (error) {
    console.log('   ⚠️  Direct API connection not available (expected in container environment)');
  }
  
  // Summary
  console.log('\n📊 Test Summary');
  console.log('===============');
  console.log(`Total Tests: ${totalTests}`);
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${totalTests - passedTests}`);
  console.log(`Success Rate: ${Math.round((passedTests / totalTests) * 100)}%`);
  
  if (passedTests === totalTests) {
    console.log('\n🎉 All tests passed! Container communication is working correctly.');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed. Check the configuration and container status.');
    process.exit(1);
  }
}

// Handle command line arguments
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`
Container Communication Test Script

Usage: node test-container-communication.js [options]

Options:
  --help, -h     Show this help message
  --frontend-port PORT  Override frontend port (default: 80)
  --api-port PORT       Override API port (default: 8000)
  --api-host HOST       Override API host (default: qwen-api)

Examples:
  node test-container-communication.js
  node test-container-communication.js --frontend-port 3000
  node test-container-communication.js --api-host localhost --api-port 8000
`);
  process.exit(0);
}

// Parse command line arguments
const frontendPortIndex = process.argv.indexOf('--frontend-port');
if (frontendPortIndex !== -1 && process.argv[frontendPortIndex + 1]) {
  TEST_CONFIG.frontend.port = parseInt(process.argv[frontendPortIndex + 1], 10);
}

const apiPortIndex = process.argv.indexOf('--api-port');
if (apiPortIndex !== -1 && process.argv[apiPortIndex + 1]) {
  TEST_CONFIG.api.port = parseInt(process.argv[apiPortIndex + 1], 10);
}

const apiHostIndex = process.argv.indexOf('--api-host');
if (apiHostIndex !== -1 && process.argv[apiHostIndex + 1]) {
  TEST_CONFIG.api.host = process.argv[apiHostIndex + 1];
}

// Run the tests
runTests().catch(error => {
  console.error('❌ Test suite failed:', error);
  process.exit(1);
});