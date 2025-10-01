#!/usr/bin/env node
/**
 * Complete Frontend-Backend Integration Test
 * Tests the full application stack including Docker containers
 */

const { spawn, exec } = require('child_process');
const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

// Configuration
const config = {
  docker: {
    composeFile: 'docker-compose.yml',
    services: ['frontend', 'api'],
    timeout: 120000 // 2 minutes for container startup
  },
  endpoints: {
    frontend: 'http://localhost:80',
    backend: 'http://localhost:8000',
    apiProxy: 'http://localhost:80/api'
  },
  tests: {
    timeout: 30000,
    retries: 5,
    retryDelay: 3000
  }
};

// Test state
const testState = {
  containersRunning: false,
  frontendHealthy: false,
  backendHealthy: false,
  integrationWorking: false,
  results: []
};

// Utility functions
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const log = (message, type = 'info') => {
  const timestamp = new Date().toISOString();
  const prefix = {
    info: '📋',
    success: '✅',
    error: '❌',
    warning: '⚠️',
    debug: '🔍'
  }[type] || '📋';
  
  console.log(`${prefix} [${timestamp}] ${message}`);
};

const execCommand = (command, options = {}) => {
  return new Promise((resolve, reject) => {
    exec(command, { timeout: 30000, ...options }, (error, stdout, stderr) => {
      if (error) {
        reject({ error, stdout, stderr });
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
};

const makeRequest = async (url, options = {}) => {
  return new Promise((resolve) => {
    try {
      const urlObj = new URL(url);
      const isHttps = urlObj.protocol === 'https:';
      const client = isHttps ? https : http;
      
      const requestOptions = {
        hostname: urlObj.hostname,
        port: urlObj.port || (isHttps ? 443 : 80),
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {},
        timeout: config.tests.timeout
      };
      
      const req = client.request(requestOptions, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          let parsedData;
          try {
            parsedData = JSON.parse(data);
          } catch (e) {
            parsedData = data;
          }
          
          resolve({
            success: true,
            status: res.statusCode,
            data: parsedData,
            headers: res.headers
          });
        });
      });
      
      req.on('error', (error) => {
        resolve({
          success: false,
          error: error.message,
          code: error.code
        });
      });
      
      req.on('timeout', () => {
        req.destroy();
        resolve({
          success: false,
          error: 'Request timeout',
          code: 'TIMEOUT'
        });
      });
      
      if (options.data) {
        req.write(JSON.stringify(options.data));
      }
      
      req.end();
    } catch (error) {
      resolve({
        success: false,
        error: error.message,
        code: error.code
      });
    }
  });
};

const retryRequest = async (url, options = {}, maxRetries = config.tests.retries) => {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const result = await makeRequest(url, options);
    
    if (result.success && result.status < 500) {
      return result;
    }
    
    if (attempt < maxRetries) {
      log(`Retry ${attempt}/${maxRetries} for ${url}`, 'debug');
      await sleep(config.tests.retryDelay);
    }
  }
  
  return await makeRequest(url, options);
};

// Docker management functions
const checkDockerStatus = async () => {
  try {
    log('Checking Docker status...');
    const result = await execCommand('docker --version');
    log(`Docker version: ${result.stdout.trim()}`, 'success');
    return true;
  } catch (error) {
    log(`Docker not available: ${error.error.message}`, 'error');
    return false;
  }
};

const checkContainerStatus = async () => {
  try {
    log('Checking container status...');
    const result = await execCommand('docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"');
    
    const containers = result.stdout.split('\n').slice(1).filter(line => line.trim());
    const qwenContainers = containers.filter(line => 
      line.includes('qwen-frontend') || line.includes('qwen-api') || line.includes('qwen-traefik')
    );
    
    log(`Found ${qwenContainers.length} Qwen containers running:`);
    qwenContainers.forEach(container => {
      log(`  ${container}`, 'info');
    });
    
    const frontendRunning = qwenContainers.some(c => c.includes('qwen-frontend'));
    const backendRunning = qwenContainers.some(c => c.includes('qwen-api'));
    
    testState.containersRunning = frontendRunning && backendRunning;
    
    return {
      frontend: frontendRunning,
      backend: backendRunning,
      total: qwenContainers.length
    };
  } catch (error) {
    log(`Failed to check container status: ${error.error.message}`, 'error');
    return { frontend: false, backend: false, total: 0 };
  }
};

const startContainers = async () => {
  try {
    log('Starting containers with docker-compose...');
    
    // Start containers in detached mode
    const startCommand = `docker-compose -f ${config.docker.composeFile} up -d --build`;
    log(`Executing: ${startCommand}`);
    
    const result = await execCommand(startCommand, { timeout: config.docker.timeout });
    log('Containers started successfully', 'success');
    
    // Wait for containers to be ready
    log('Waiting for containers to initialize...');
    await sleep(10000); // Initial wait
    
    return true;
  } catch (error) {
    log(`Failed to start containers: ${error.error.message}`, 'error');
    log(`STDOUT: ${error.stdout}`, 'debug');
    log(`STDERR: ${error.stderr}`, 'debug');
    return false;
  }
};

// Health check functions
const testFrontendHealth = async () => {
  log('Testing frontend health...');
  
  const healthUrl = `${config.endpoints.frontend}/health`;
  const result = await retryRequest(healthUrl);
  
  if (result.success && result.status === 200) {
    log('Frontend health check passed', 'success');
    testState.frontendHealthy = true;
    testState.results.push({
      test: 'Frontend Health',
      status: 'PASS',
      details: result.data
    });
    return true;
  } else {
    log(`Frontend health check failed: ${result.error || `HTTP ${result.status}`}`, 'error');
    testState.results.push({
      test: 'Frontend Health',
      status: 'FAIL',
      error: result.error || `HTTP ${result.status}`,
      details: result.data
    });
    return false;
  }
};

const testBackendHealth = async () => {
  log('Testing backend health...');
  
  const healthUrl = `${config.endpoints.backend}/health`;
  const result = await retryRequest(healthUrl);
  
  if (result.success && result.status === 200) {
    log('Backend health check passed', 'success');
    testState.backendHealthy = true;
    testState.results.push({
      test: 'Backend Health',
      status: 'PASS',
      details: result.data
    });
    return true;
  } else {
    log(`Backend health check failed: ${result.error || `HTTP ${result.status}`}`, 'error');
    testState.results.push({
      test: 'Backend Health',
      status: 'FAIL',
      error: result.error || `HTTP ${result.status}`,
      details: result.data
    });
    return false;
  }
};

const testApiProxy = async () => {
  log('Testing API proxy communication...');
  
  const proxyUrl = `${config.endpoints.apiProxy}/health`;
  const result = await retryRequest(proxyUrl);
  
  if (result.success && result.status === 200) {
    log('API proxy communication working', 'success');
    testState.results.push({
      test: 'API Proxy',
      status: 'PASS',
      details: result.data
    });
    return true;
  } else {
    log(`API proxy failed: ${result.error || `HTTP ${result.status}`}`, 'error');
    testState.results.push({
      test: 'API Proxy',
      status: 'FAIL',
      error: result.error || `HTTP ${result.status}`,
      details: result.data
    });
    return false;
  }
};

const testBackendStatus = async () => {
  log('Testing backend status endpoint...');
  
  const statusUrl = `${config.endpoints.apiProxy}/status`;
  const result = await retryRequest(statusUrl);
  
  if (result.success && result.status === 200) {
    const status = result.data;
    log(`Backend status: ${status.initialization?.status || 'unknown'}`, 'success');
    log(`Model loaded: ${status.model_loaded}`, 'info');
    log(`Device: ${status.device}`, 'info');
    
    testState.results.push({
      test: 'Backend Status',
      status: 'PASS',
      details: {
        modelLoaded: status.model_loaded,
        device: status.device,
        initStatus: status.initialization?.status
      }
    });
    return true;
  } else {
    log(`Backend status check failed: ${result.error || `HTTP ${result.status}`}`, 'error');
    testState.results.push({
      test: 'Backend Status',
      status: 'FAIL',
      error: result.error || `HTTP ${result.status}`
    });
    return false;
  }
};

const testAspectRatios = async () => {
  log('Testing aspect ratios API...');
  
  const ratiosUrl = `${config.endpoints.apiProxy}/aspect-ratios`;
  const result = await retryRequest(ratiosUrl);
  
  if (result.success && result.status === 200 && result.data?.ratios) {
    const ratioCount = Object.keys(result.data.ratios).length;
    log(`Retrieved ${ratioCount} aspect ratios`, 'success');
    
    testState.results.push({
      test: 'Aspect Ratios API',
      status: 'PASS',
      details: { ratioCount }
    });
    return true;
  } else {
    log(`Aspect ratios API failed: ${result.error || `HTTP ${result.status}`}`, 'error');
    testState.results.push({
      test: 'Aspect Ratios API',
      status: 'FAIL',
      error: result.error || `HTTP ${result.status}`
    });
    return false;
  }
};

const testCorsHeaders = async () => {
  log('Testing CORS headers...');
  
  const corsUrl = `${config.endpoints.apiProxy}/health`;
  const result = await retryRequest(corsUrl, {
    headers: {
      'Origin': config.endpoints.frontend,
      'Access-Control-Request-Method': 'GET'
    }
  });
  
  if (result.success) {
    const corsHeaders = {
      'access-control-allow-origin': result.headers['access-control-allow-origin'],
      'access-control-allow-methods': result.headers['access-control-allow-methods']
    };
    
    const hasCors = Object.values(corsHeaders).some(header => header !== undefined);
    
    if (hasCors) {
      log('CORS headers configured correctly', 'success');
      testState.results.push({
        test: 'CORS Headers',
        status: 'PASS',
        details: corsHeaders
      });
      return true;
    }
  }
  
  log('CORS headers missing or incorrect', 'warning');
  testState.results.push({
    test: 'CORS Headers',
    status: 'FAIL',
    error: 'Missing CORS headers'
  });
  return false;
};

const testCompleteIntegration = async () => {
  log('Testing complete frontend-backend integration...');
  
  // Test multiple endpoints to ensure full integration
  const endpoints = [
    '/health',
    '/status',
    '/aspect-ratios',
    '/health/detailed'
  ];
  
  let successCount = 0;
  const endpointResults = [];
  
  for (const endpoint of endpoints) {
    const url = `${config.endpoints.apiProxy}${endpoint}`;
    const result = await retryRequest(url);
    
    const success = result.success && result.status === 200;
    if (success) successCount++;
    
    endpointResults.push({
      endpoint,
      success,
      status: result.status,
      error: result.error
    });
  }
  
  const integrationWorking = successCount >= endpoints.length * 0.75; // 75% success rate
  testState.integrationWorking = integrationWorking;
  
  if (integrationWorking) {
    log(`Integration test passed: ${successCount}/${endpoints.length} endpoints working`, 'success');
    testState.results.push({
      test: 'Complete Integration',
      status: 'PASS',
      details: { successCount, totalEndpoints: endpoints.length, endpoints: endpointResults }
    });
  } else {
    log(`Integration test failed: only ${successCount}/${endpoints.length} endpoints working`, 'error');
    testState.results.push({
      test: 'Complete Integration',
      status: 'FAIL',
      details: { successCount, totalEndpoints: endpoints.length, endpoints: endpointResults }
    });
  }
  
  return integrationWorking;
};

// Main test execution
const runIntegrationTest = async () => {
  console.log('🚀 Frontend-Backend Integration Test Suite');
  console.log('='.repeat(60));
  
  try {
    // Step 1: Check Docker
    const dockerAvailable = await checkDockerStatus();
    if (!dockerAvailable) {
      log('Docker is required but not available', 'error');
      process.exit(1);
    }
    
    // Step 2: Check current container status
    const containerStatus = await checkContainerStatus();
    
    if (!containerStatus.frontend || !containerStatus.backend) {
      log('Required containers not running, attempting to start them...', 'warning');
      const started = await startContainers();
      
      if (!started) {
        log('Failed to start containers', 'error');
        process.exit(1);
      }
      
      // Recheck container status
      await sleep(5000);
      const newStatus = await checkContainerStatus();
      if (!newStatus.frontend || !newStatus.backend) {
        log('Containers still not running after start attempt', 'error');
        process.exit(1);
      }
    }
    
    // Step 3: Wait for services to be ready
    log('Waiting for services to initialize...');
    await sleep(15000);
    
    // Step 4: Run health checks
    const frontendHealthy = await testFrontendHealth();
    const backendHealthy = await testBackendHealth();
    
    if (!frontendHealthy || !backendHealthy) {
      log('Basic health checks failed', 'error');
    }
    
    // Step 5: Test API communication
    const proxyWorking = await testApiProxy();
    if (!proxyWorking) {
      log('API proxy communication failed', 'error');
    }
    
    // Step 6: Test specific endpoints
    await testBackendStatus();
    await testAspectRatios();
    await testCorsHeaders();
    
    // Step 7: Complete integration test
    await testCompleteIntegration();
    
    // Generate summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 Integration Test Results');
    console.log('='.repeat(60));
    
    const passedTests = testState.results.filter(r => r.status === 'PASS').length;
    const totalTests = testState.results.length;
    const successRate = totalTests > 0 ? ((passedTests / totalTests) * 100).toFixed(1) : 0;
    
    console.log(`✅ Passed: ${passedTests}/${totalTests} (${successRate}%)`);
    console.log(`🐳 Containers Running: ${testState.containersRunning ? 'Yes' : 'No'}`);
    console.log(`🌐 Frontend Healthy: ${testState.frontendHealthy ? 'Yes' : 'No'}`);
    console.log(`🔧 Backend Healthy: ${testState.backendHealthy ? 'Yes' : 'No'}`);
    console.log(`🔗 Integration Working: ${testState.integrationWorking ? 'Yes' : 'No'}`);
    
    // Detailed results
    console.log('\n📋 Detailed Test Results:');
    testState.results.forEach(result => {
      const icon = result.status === 'PASS' ? '✅' : '❌';
      console.log(`${icon} ${result.test}: ${result.status}`);
      if (result.error) {
        console.log(`   Error: ${result.error}`);
      }
    });
    
    // Save results to file
    const reportPath = path.join(__dirname, 'integration-test-report.json');
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        passedTests,
        totalTests,
        successRate: parseFloat(successRate),
        containersRunning: testState.containersRunning,
        frontendHealthy: testState.frontendHealthy,
        backendHealthy: testState.backendHealthy,
        integrationWorking: testState.integrationWorking
      },
      config,
      results: testState.results
    };
    
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
    
    // Final status
    if (testState.integrationWorking && testState.frontendHealthy && testState.backendHealthy) {
      console.log('\n🎉 Frontend-Backend integration is working correctly!');
      console.log(`\n🌐 Access the application at: ${config.endpoints.frontend}`);
      console.log(`🔧 Backend API available at: ${config.endpoints.backend}`);
      console.log(`🔗 API proxy working at: ${config.endpoints.apiProxy}`);
      process.exit(0);
    } else {
      console.log('\n❌ Integration test failed - issues detected');
      process.exit(1);
    }
    
  } catch (error) {
    log(`Integration test failed with error: ${error.message}`, 'error');
    console.error(error);
    process.exit(1);
  }
};

// Handle command line arguments
if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`
Frontend-Backend Integration Test Suite

Usage: node test-frontend-backend-integration.js [options]

Options:
  --no-start    Don't attempt to start containers
  --verbose     Show detailed output
  --help, -h    Show this help message

This script will:
1. Check if Docker is available
2. Verify containers are running (start them if needed)
3. Test frontend and backend health
4. Verify API proxy communication
5. Test specific API endpoints
6. Generate a comprehensive report
`);
  process.exit(0);
}

// Run the integration test
if (require.main === module) {
  runIntegrationTest().catch(error => {
    console.error('❌ Integration test runner failed:', error);
    process.exit(1);
  });
}

module.exports = { runIntegrationTest, testState };