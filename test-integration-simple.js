#!/usr/bin/env node
/**
 * Simple Frontend-Backend Integration Test
 * Tests the integration between the React frontend and existing backend
 */

const http = require('http');
const { exec } = require('child_process');
const fs = require('fs');

// Configuration
const config = {
  backend: {
    url: 'http://localhost:3000',
    healthEndpoint: '/health',
    statusEndpoint: '/status'
  },
  frontend: {
    url: 'http://localhost:8080',
    healthEndpoint: '/health'
  },
  apiProxy: {
    url: 'http://localhost:8080/api',
    healthEndpoint: '/health'
  },
  timeout: 30000
};

// Test results
const results = {
  tests: [],
  passed: 0,
  failed: 0
};

// Utility functions
const log = (message, type = 'info') => {
  const timestamp = new Date().toISOString();
  const icons = { info: '📋', success: '✅', error: '❌', warning: '⚠️' };
  console.log(`${icons[type]} [${timestamp}] ${message}`);
};

const makeRequest = (url) => {
  return new Promise((resolve) => {
    const request = http.get(url, { timeout: config.timeout }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ success: true, status: res.statusCode, data: parsed });
        } catch (e) {
          resolve({ success: true, status: res.statusCode, data: data });
        }
      });
    });

    request.on('error', (error) => {
      resolve({ success: false, error: error.message });
    });

    request.on('timeout', () => {
      request.destroy();
      resolve({ success: false, error: 'Request timeout' });
    });
  });
};

const execCommand = (command) => {
  return new Promise((resolve, reject) => {
    exec(command, { timeout: 120000 }, (error, stdout, stderr) => {
      if (error) {
        reject({ error, stdout, stderr });
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
};

const addTestResult = (name, success, message, details = null) => {
  results.tests.push({ name, success, message, details, timestamp: new Date().toISOString() });
  if (success) {
    results.passed++;
    log(`${name}: ${message}`, 'success');
  } else {
    results.failed++;
    log(`${name}: ${message}`, 'error');
  }
};

// Test functions
const testBackendHealth = async () => {
  log('Testing backend health...');
  const result = await makeRequest(`${config.backend.url}${config.backend.healthEndpoint}`);
  
  if (result.success && result.status === 200) {
    addTestResult('Backend Health', true, 'Backend is healthy and responding', result.data);
    return true;
  } else {
    addTestResult('Backend Health', false, `Backend health check failed: ${result.error || result.status}`, result);
    return false;
  }
};

const testBackendStatus = async () => {
  log('Testing backend status...');
  const result = await makeRequest(`${config.backend.url}${config.backend.statusEndpoint}`);
  
  if (result.success && result.status === 200) {
    const status = result.data;
    addTestResult('Backend Status', true, `Status: ${status.initialization?.status || 'ready'}`, {
      modelLoaded: status.model_loaded,
      device: status.device
    });
    return true;
  } else {
    addTestResult('Backend Status', false, `Backend status check failed: ${result.error || result.status}`, result);
    return false;
  }
};

const buildFrontend = async () => {
  log('Building frontend container...');
  try {
    // Build the frontend container using the integration Dockerfile
    const buildCommand = 'docker build -f Dockerfile.integration -t qwen-frontend:integration-test .';
    const result = await execCommand(buildCommand);
    
    addTestResult('Frontend Build', true, 'Frontend container built successfully');
    return true;
  } catch (error) {
    addTestResult('Frontend Build', false, `Frontend build failed: ${error.error?.message || error.message}`, {
      stdout: error.stdout,
      stderr: error.stderr
    });
    return false;
  }
};

const startFrontend = async () => {
  log('Starting frontend container...');
  try {
    // Stop any existing frontend container
    try {
      await execCommand('docker stop qwen-frontend-test 2>/dev/null || true');
      await execCommand('docker rm qwen-frontend-test 2>/dev/null || true');
    } catch (e) {
      // Ignore errors
    }
    
    // Start the frontend container
    const startCommand = `docker run -d --name qwen-frontend-test -p 8080:80 --add-host host.docker.internal:host-gateway qwen-frontend:integration-test`;
    await execCommand(startCommand);
    
    // Wait for container to start
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    addTestResult('Frontend Start', true, 'Frontend container started successfully');
    return true;
  } catch (error) {
    addTestResult('Frontend Start', false, `Frontend start failed: ${error.error?.message || error.message}`, {
      stdout: error.stdout,
      stderr: error.stderr
    });
    return false;
  }
};

const testFrontendHealth = async () => {
  log('Testing frontend health...');
  
  // Retry logic for frontend health check
  for (let i = 0; i < 5; i++) {
    const result = await makeRequest(`${config.frontend.url}${config.frontend.healthEndpoint}`);
    
    if (result.success && result.status === 200) {
      addTestResult('Frontend Health', true, 'Frontend is healthy and serving content', result.data);
      return true;
    }
    
    if (i < 4) {
      log(`Frontend health check attempt ${i + 1} failed, retrying...`, 'warning');
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  }
  
  addTestResult('Frontend Health', false, 'Frontend health check failed after retries');
  return false;
};

const testApiProxy = async () => {
  log('Testing API proxy...');
  const result = await makeRequest(`${config.apiProxy.url}${config.apiProxy.healthEndpoint}`);
  
  if (result.success && result.status === 200) {
    addTestResult('API Proxy', true, 'API proxy is working correctly', result.data);
    return true;
  } else {
    addTestResult('API Proxy', false, `API proxy failed: ${result.error || result.status}`, result);
    return false;
  }
};

const testFrontendBackendIntegration = async () => {
  log('Testing complete frontend-backend integration...');
  
  // Test multiple endpoints through the proxy
  const endpoints = ['/health', '/status', '/aspect-ratios'];
  let successCount = 0;
  
  for (const endpoint of endpoints) {
    const result = await makeRequest(`${config.apiProxy.url}${endpoint}`);
    if (result.success && result.status === 200) {
      successCount++;
    }
  }
  
  const integrationWorking = successCount >= 2; // At least 2/3 endpoints working
  
  if (integrationWorking) {
    addTestResult('Integration Test', true, `${successCount}/${endpoints.length} endpoints working through proxy`, {
      successCount,
      totalEndpoints: endpoints.length
    });
    return true;
  } else {
    addTestResult('Integration Test', false, `Only ${successCount}/${endpoints.length} endpoints working`, {
      successCount,
      totalEndpoints: endpoints.length
    });
    return false;
  }
};

const cleanup = async () => {
  log('Cleaning up test containers...');
  try {
    await execCommand('docker stop qwen-frontend-test 2>/dev/null || true');
    await execCommand('docker rm qwen-frontend-test 2>/dev/null || true');
    log('Cleanup completed', 'success');
  } catch (error) {
    log('Cleanup failed, but continuing...', 'warning');
  }
};

// Main test execution
const runIntegrationTest = async () => {
  console.log('🚀 Frontend-Backend Integration Test');
  console.log('='.repeat(50));
  
  try {
    // Step 1: Test backend
    const backendHealthy = await testBackendHealth();
    if (!backendHealthy) {
      log('Backend is not healthy, cannot proceed with integration test', 'error');
      process.exit(1);
    }
    
    await testBackendStatus();
    
    // Step 2: Build and start frontend
    const frontendBuilt = await buildFrontend();
    if (!frontendBuilt) {
      log('Frontend build failed, cannot proceed', 'error');
      process.exit(1);
    }
    
    const frontendStarted = await startFrontend();
    if (!frontendStarted) {
      log('Frontend start failed, cannot proceed', 'error');
      process.exit(1);
    }
    
    // Step 3: Test frontend
    const frontendHealthy = await testFrontendHealth();
    
    // Step 4: Test integration
    const proxyWorking = await testApiProxy();
    const integrationWorking = await testFrontendBackendIntegration();
    
    // Generate summary
    console.log('\n' + '='.repeat(50));
    console.log('📊 Integration Test Results');
    console.log('='.repeat(50));
    console.log(`✅ Passed: ${results.passed}`);
    console.log(`❌ Failed: ${results.failed}`);
    console.log(`📈 Success Rate: ${((results.passed / (results.passed + results.failed)) * 100).toFixed(1)}%`);
    
    // Detailed results
    console.log('\n📋 Test Details:');
    results.tests.forEach(test => {
      const icon = test.success ? '✅' : '❌';
      console.log(`${icon} ${test.name}: ${test.message}`);
    });
    
    // Save results
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        passed: results.passed,
        failed: results.failed,
        successRate: (results.passed / (results.passed + results.failed)) * 100
      },
      tests: results.tests
    };
    
    fs.writeFileSync('integration-test-results.json', JSON.stringify(report, null, 2));
    console.log('\n📄 Results saved to integration-test-results.json');
    
    // Final status
    if (integrationWorking && frontendHealthy && proxyWorking) {
      console.log('\n🎉 Frontend-Backend integration is working!');
      console.log(`🌐 Frontend: ${config.frontend.url}`);
      console.log(`🔧 Backend: ${config.backend.url}`);
      console.log(`🔗 API Proxy: ${config.apiProxy.url}`);
      
      if (process.argv.includes('--keep-running')) {
        console.log('\n⏳ Keeping containers running for manual testing...');
        console.log('Press Ctrl+C to stop and cleanup');
        process.on('SIGINT', async () => {
          await cleanup();
          process.exit(0);
        });
        // Keep the process alive
        setInterval(() => {}, 1000);
      } else {
        await cleanup();
        process.exit(0);
      }
    } else {
      console.log('\n❌ Integration test failed');
      await cleanup();
      process.exit(1);
    }
    
  } catch (error) {
    console.error('❌ Test execution failed:', error);
    await cleanup();
    process.exit(1);
  }
};

// Handle command line arguments
if (process.argv.includes('--help')) {
  console.log(`
Frontend-Backend Integration Test

Usage: node test-integration-simple.js [options]

Options:
  --keep-running    Keep containers running after test for manual inspection
  --help           Show this help message

This script will:
1. Test backend health and status
2. Build the frontend container
3. Start the frontend container
4. Test frontend health
5. Test API proxy communication
6. Test complete integration
7. Generate a report
`);
  process.exit(0);
}

// Run the test
runIntegrationTest();