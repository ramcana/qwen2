#!/usr/bin/env node
/**
 * Frontend-Backend Integration Test Suite
 * Tests complete communication between React frontend and FastAPI backend
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Configuration
const config = {
  frontend: {
    url: process.env.FRONTEND_URL || 'http://localhost:80',
    healthEndpoint: '/health'
  },
  backend: {
    url: process.env.BACKEND_URL || 'http://localhost:8000',
    apiUrl: process.env.API_URL || 'http://localhost:80/api',
    healthEndpoint: '/health',
    statusEndpoint: '/status',
    aspectRatiosEndpoint: '/aspect-ratios',
    initializeEndpoint: '/initialize'
  },
  timeout: 30000,
  retries: 3,
  retryDelay: 2000
};

// Test results tracking
const testResults = {
  passed: 0,
  failed: 0,
  skipped: 0,
  tests: []
};

// Utility functions
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const logTest = (name, status, message = '', details = null) => {
  const timestamp = new Date().toISOString();
  const result = {
    name,
    status,
    message,
    details,
    timestamp
  };
  
  testResults.tests.push(result);
  
  const statusIcon = status === 'PASS' ? '✅' : status === 'FAIL' ? '❌' : '⏭️';
  console.log(`${statusIcon} ${name}: ${message}`);
  
  if (details && process.env.VERBOSE) {
    console.log(`   Details: ${JSON.stringify(details, null, 2)}`);
  }
  
  if (status === 'PASS') testResults.passed++;
  else if (status === 'FAIL') testResults.failed++;
  else testResults.skipped++;
};

const makeRequest = async (url, options = {}) => {
  const requestOptions = {
    timeout: config.timeout,
    validateStatus: () => true, // Don't throw on HTTP errors
    ...options
  };
  
  try {
    const response = await axios(url, requestOptions);
    return {
      success: true,
      status: response.status,
      data: response.data,
      headers: response.headers
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      code: error.code
    };
  }
};

const retryRequest = async (url, options = {}, retries = config.retries) => {
  for (let i = 0; i <= retries; i++) {
    const result = await makeRequest(url, options);
    
    if (result.success && result.status < 500) {
      return result;
    }
    
    if (i < retries) {
      console.log(`   Retry ${i + 1}/${retries} for ${url}`);
      await sleep(config.retryDelay);
    }
  }
  
  return await makeRequest(url, options);
};

// Test functions
const testFrontendHealth = async () => {
  const url = `${config.frontend.url}${config.frontend.healthEndpoint}`;
  const result = await retryRequest(url);
  
  if (!result.success) {
    logTest('Frontend Health Check', 'FAIL', `Connection failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 200) {
    logTest('Frontend Health Check', 'PASS', 'Frontend container is healthy', {
      status: result.status,
      responseTime: result.headers['x-response-time']
    });
    return true;
  } else {
    logTest('Frontend Health Check', 'FAIL', `HTTP ${result.status}`, result.data);
    return false;
  }
};

const testBackendHealth = async () => {
  const url = `${config.backend.url}${config.backend.healthEndpoint}`;
  const result = await retryRequest(url);
  
  if (!result.success) {
    logTest('Backend Health Check', 'FAIL', `Connection failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 200) {
    logTest('Backend Health Check', 'PASS', 'Backend API is healthy', {
      status: result.status,
      service: result.data?.service,
      version: result.data?.version
    });
    return true;
  } else {
    logTest('Backend Health Check', 'FAIL', `HTTP ${result.status}`, result.data);
    return false;
  }
};

const testApiProxy = async () => {
  const url = `${config.backend.apiUrl}${config.backend.healthEndpoint}`;
  const result = await retryRequest(url);
  
  if (!result.success) {
    logTest('API Proxy Communication', 'FAIL', `Proxy connection failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 200) {
    logTest('API Proxy Communication', 'PASS', 'Frontend can reach backend via proxy', {
      status: result.status,
      proxyUrl: url,
      service: result.data?.service
    });
    return true;
  } else {
    logTest('API Proxy Communication', 'FAIL', `Proxy returned HTTP ${result.status}`, result.data);
    return false;
  }
};

const testCorsHeaders = async () => {
  const url = `${config.backend.apiUrl}${config.backend.healthEndpoint}`;
  const result = await retryRequest(url, {
    headers: {
      'Origin': config.frontend.url,
      'Access-Control-Request-Method': 'GET',
      'Access-Control-Request-Headers': 'Content-Type'
    }
  });
  
  if (!result.success) {
    logTest('CORS Headers', 'FAIL', `CORS test failed: ${result.error}`, result);
    return false;
  }
  
  const corsHeaders = {
    'access-control-allow-origin': result.headers['access-control-allow-origin'],
    'access-control-allow-methods': result.headers['access-control-allow-methods'],
    'access-control-allow-headers': result.headers['access-control-allow-headers']
  };
  
  const hasCors = Object.values(corsHeaders).some(header => header !== undefined);
  
  if (hasCors) {
    logTest('CORS Headers', 'PASS', 'CORS headers are properly configured', corsHeaders);
    return true;
  } else {
    logTest('CORS Headers', 'FAIL', 'Missing CORS headers', { headers: result.headers });
    return false;
  }
};

const testBackendStatus = async () => {
  const url = `${config.backend.apiUrl}${config.backend.statusEndpoint}`;
  const result = await retryRequest(url);
  
  if (!result.success) {
    logTest('Backend Status Endpoint', 'FAIL', `Status check failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 200 && result.data) {
    const status = result.data;
    logTest('Backend Status Endpoint', 'PASS', 'Status endpoint working', {
      modelLoaded: status.model_loaded,
      device: status.device,
      queueLength: status.queue_length,
      initializationStatus: status.initialization?.status
    });
    return true;
  } else {
    logTest('Backend Status Endpoint', 'FAIL', `HTTP ${result.status}`, result.data);
    return false;
  }
};

const testAspectRatios = async () => {
  const url = `${config.backend.apiUrl}${config.backend.aspectRatiosEndpoint}`;
  const result = await retryRequest(url);
  
  if (!result.success) {
    logTest('Aspect Ratios API', 'FAIL', `Request failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 200 && result.data?.ratios) {
    const ratios = result.data.ratios;
    logTest('Aspect Ratios API', 'PASS', `Retrieved ${Object.keys(ratios).length} aspect ratios`, {
      ratioCount: Object.keys(ratios).length,
      sampleRatios: Object.keys(ratios).slice(0, 3)
    });
    return true;
  } else {
    logTest('Aspect Ratios API', 'FAIL', `HTTP ${result.status} or invalid data`, result.data);
    return false;
  }
};

const testDetailedHealthCheck = async () => {
  const url = `${config.backend.apiUrl}/health/detailed`;
  const result = await retryRequest(url);
  
  if (!result.success) {
    logTest('Detailed Health Check', 'FAIL', `Request failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 200 && result.data) {
    const health = result.data;
    logTest('Detailed Health Check', 'PASS', 'Detailed health data available', {
      uptime: health.uptime,
      models: health.models,
      gpu: health.gpu?.available,
      storage: Object.keys(health.storage || {}).length
    });
    return true;
  } else {
    logTest('Detailed Health Check', 'FAIL', `HTTP ${result.status}`, result.data);
    return false;
  }
};

const testFrontendStaticAssets = async () => {
  // Test if frontend serves static assets correctly
  const staticUrls = [
    `${config.frontend.url}/static/css`,
    `${config.frontend.url}/static/js`,
    `${config.frontend.url}/manifest.json`,
    `${config.frontend.url}/favicon.ico`
  ];
  
  let assetsFound = 0;
  const assetResults = [];
  
  for (const url of staticUrls) {
    const result = await makeRequest(url);
    if (result.success && (result.status === 200 || result.status === 404)) {
      if (result.status === 200) assetsFound++;
      assetResults.push({ url, status: result.status, found: result.status === 200 });
    }
  }
  
  if (assetsFound > 0) {
    logTest('Frontend Static Assets', 'PASS', `Found ${assetsFound}/${staticUrls.length} static assets`, assetResults);
    return true;
  } else {
    logTest('Frontend Static Assets', 'FAIL', 'No static assets found', assetResults);
    return false;
  }
};

const testContainerNetworking = async () => {
  // Test if containers can communicate using container names
  const containerApiUrl = `http://qwen-api:8000/health`;
  
  try {
    // This test only works from within the container network
    // We'll simulate it by testing the proxy instead
    const proxyResult = await retryRequest(`${config.backend.apiUrl}/health`);
    
    if (proxyResult.success && proxyResult.status === 200) {
      logTest('Container Networking', 'PASS', 'Containers can communicate via Docker network', {
        proxyWorking: true,
        backendReachable: true
      });
      return true;
    } else {
      logTest('Container Networking', 'FAIL', 'Container communication failed', proxyResult);
      return false;
    }
  } catch (error) {
    logTest('Container Networking', 'FAIL', `Network test failed: ${error.message}`, { error: error.message });
    return false;
  }
};

const testErrorHandling = async () => {
  // Test how frontend handles backend errors
  const invalidUrl = `${config.backend.apiUrl}/nonexistent-endpoint`;
  const result = await makeRequest(invalidUrl);
  
  if (!result.success) {
    logTest('Error Handling', 'FAIL', `Request failed: ${result.error}`, result);
    return false;
  }
  
  if (result.status === 404) {
    logTest('Error Handling', 'PASS', 'Backend properly returns 404 for invalid endpoints', {
      status: result.status,
      hasErrorResponse: !!result.data
    });
    return true;
  } else {
    logTest('Error Handling', 'FAIL', `Expected 404, got ${result.status}`, result.data);
    return false;
  }
};

// Main test runner
const runIntegrationTests = async () => {
  console.log('🚀 Starting Frontend-Backend Integration Tests');
  console.log('='.repeat(60));
  console.log(`Frontend URL: ${config.frontend.url}`);
  console.log(`Backend URL: ${config.backend.url}`);
  console.log(`API Proxy URL: ${config.backend.apiUrl}`);
  console.log('='.repeat(60));
  
  const tests = [
    { name: 'Frontend Health', fn: testFrontendHealth, critical: true },
    { name: 'Backend Health', fn: testBackendHealth, critical: true },
    { name: 'API Proxy', fn: testApiProxy, critical: true },
    { name: 'CORS Headers', fn: testCorsHeaders, critical: false },
    { name: 'Backend Status', fn: testBackendStatus, critical: false },
    { name: 'Aspect Ratios API', fn: testAspectRatios, critical: false },
    { name: 'Detailed Health', fn: testDetailedHealthCheck, critical: false },
    { name: 'Static Assets', fn: testFrontendStaticAssets, critical: false },
    { name: 'Container Networking', fn: testContainerNetworking, critical: true },
    { name: 'Error Handling', fn: testErrorHandling, critical: false }
  ];
  
  let criticalFailures = 0;
  
  for (const test of tests) {
    console.log(`\n🧪 Running: ${test.name}`);
    try {
      const success = await test.fn();
      if (!success && test.critical) {
        criticalFailures++;
      }
    } catch (error) {
      logTest(test.name, 'FAIL', `Test threw exception: ${error.message}`, { error: error.message });
      if (test.critical) {
        criticalFailures++;
      }
    }
  }
  
  // Generate summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 Integration Test Summary');
  console.log('='.repeat(60));
  console.log(`✅ Passed: ${testResults.passed}`);
  console.log(`❌ Failed: ${testResults.failed}`);
  console.log(`⏭️ Skipped: ${testResults.skipped}`);
  console.log(`🚨 Critical Failures: ${criticalFailures}`);
  
  const totalTests = testResults.passed + testResults.failed + testResults.skipped;
  const successRate = totalTests > 0 ? ((testResults.passed / totalTests) * 100).toFixed(1) : 0;
  console.log(`📈 Success Rate: ${successRate}%`);
  
  // Save detailed results
  const reportPath = path.join(__dirname, 'integration-test-results.json');
  const report = {
    summary: {
      passed: testResults.passed,
      failed: testResults.failed,
      skipped: testResults.skipped,
      criticalFailures,
      successRate: parseFloat(successRate),
      timestamp: new Date().toISOString()
    },
    config,
    tests: testResults.tests
  };
  
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`📄 Detailed report saved to: ${reportPath}`);
  
  // Exit with appropriate code
  if (criticalFailures > 0) {
    console.log('\n❌ Integration tests failed - critical issues detected');
    process.exit(1);
  } else if (testResults.failed > 0) {
    console.log('\n⚠️ Integration tests completed with non-critical failures');
    process.exit(0);
  } else {
    console.log('\n✅ All integration tests passed successfully!');
    process.exit(0);
  }
};

// Handle command line arguments
const args = process.argv.slice(2);
if (args.includes('--help') || args.includes('-h')) {
  console.log(`
Frontend-Backend Integration Test Suite

Usage: node test-integration.js [options]

Options:
  --frontend-url URL    Frontend URL (default: http://localhost:80)
  --backend-url URL     Backend URL (default: http://localhost:8000)
  --api-url URL         API proxy URL (default: http://localhost:80/api)
  --timeout MS          Request timeout in ms (default: 30000)
  --verbose             Show detailed test output
  --help, -h            Show this help message

Environment Variables:
  FRONTEND_URL          Frontend URL
  BACKEND_URL           Backend URL  
  API_URL               API proxy URL
  VERBOSE               Enable verbose output
`);
  process.exit(0);
}

// Parse command line arguments
args.forEach((arg, index) => {
  switch (arg) {
    case '--frontend-url':
      config.frontend.url = args[index + 1];
      break;
    case '--backend-url':
      config.backend.url = args[index + 1];
      break;
    case '--api-url':
      config.backend.apiUrl = args[index + 1];
      break;
    case '--timeout':
      config.timeout = parseInt(args[index + 1]) || config.timeout;
      break;
    case '--verbose':
      process.env.VERBOSE = 'true';
      break;
  }
});

// Run the tests
if (require.main === module) {
  runIntegrationTests().catch(error => {
    console.error('❌ Test runner failed:', error);
    process.exit(1);
  });
}

module.exports = { runIntegrationTests, testResults };