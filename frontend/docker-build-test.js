#!/usr/bin/env node
/**
 * Enhanced Docker Build Testing Script
 * Comprehensive testing for Docker build process and container functionality
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

class DockerBuildTester {
    constructor() {
        this.testId = `frontend-test-${Date.now()}`;
        this.testPort = 3001;
        this.results = {
            timestamp: new Date().toISOString(),
            testId: this.testId,
            status: 'unknown',
            tests: {},
            errors: [],
            warnings: [],
            metrics: {},
            containerInfo: {}
        };
        this.startTime = Date.now();
        this.cleanup = true;
    }

    /**
     * Log test step
     */
    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = {
            info: '🔍',
            success: '✅',
            warning: '⚠️',
            error: '❌',
            build: '🔨',
            test: '🧪',
            cleanup: '🧹'
        }[type] || 'ℹ️';
        
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    /**
     * Execute command with error handling
     */
    execCommand(command, options = {}) {
        try {
            const result = execSync(command, { 
                encoding: 'utf8', 
                stdio: options.silent ? 'pipe' : 'inherit',
                ...options 
            });
            return { success: true, output: result };
        } catch (error) {
            return { 
                success: false, 
                error: error.message, 
                output: error.stdout || error.stderr || '' 
            };
        }
    }

    /**
     * Check if Docker is available
     */
    checkDockerAvailability() {
        this.log('Checking Docker availability...');
        
        const result = this.execCommand('docker --version', { silent: true });
        if (!result.success) {
            throw new Error('Docker is not available or not running');
        }
        
        this.results.tests.docker_available = true;
        this.log('Docker is available', 'success');
    }

    /**
     * Check if port is available
     */
    checkPortAvailability() {
        this.log(`Checking if port ${this.testPort} is available...`);
        
        return new Promise((resolve, reject) => {
            const server = http.createServer();
            
            server.listen(this.testPort, () => {
                server.close(() => {
                    this.results.tests.port_available = true;
                    this.log(`Port ${this.testPort} is available`, 'success');
                    resolve();
                });
            });
            
            server.on('error', (error) => {
                if (error.code === 'EADDRINUSE') {
                    this.results.tests.port_available = false;
                    this.results.errors.push(`Port ${this.testPort} is already in use`);
                    reject(new Error(`Port ${this.testPort} is already in use`));
                } else {
                    reject(error);
                }
            });
        });
    }

    /**
     * Build Docker image
     */
    async buildDockerImage() {
        this.log('Building Docker image...', 'build');
        
        const buildStartTime = Date.now();
        
        // Clean up any existing test images
        this.execCommand(`docker rmi ${this.testId} 2>/dev/null || true`, { silent: true });
        
        const buildCommand = `docker build -f Dockerfile.prod -t ${this.testId} .`;
        const result = this.execCommand(buildCommand);
        
        const buildTime = Date.now() - buildStartTime;
        this.results.metrics.build_time_ms = buildTime;
        
        if (!result.success) {
            this.results.tests.docker_build = false;
            this.results.errors.push(`Docker build failed: ${result.error}`);
            throw new Error(`Docker build failed: ${result.error}`);
        }
        
        // Get image information
        const imageInfo = this.execCommand(`docker inspect ${this.testId}`, { silent: true });
        if (imageInfo.success) {
            try {
                const info = JSON.parse(imageInfo.output)[0];
                this.results.containerInfo.imageId = info.Id;
                this.results.containerInfo.imageSize = info.Size;
                this.results.containerInfo.created = info.Created;
            } catch (error) {
                this.log('Could not parse image info', 'warning');
            }
        }
        
        this.results.tests.docker_build = true;
        this.log(`Docker build completed successfully (${buildTime}ms)`, 'success');
    }

    /**
     * Start container
     */
    async startContainer() {
        this.log('Starting container...', 'test');
        
        // Stop any existing test container
        this.execCommand(`docker stop ${this.testId} 2>/dev/null || true`, { silent: true });
        this.execCommand(`docker rm ${this.testId} 2>/dev/null || true`, { silent: true });
        
        const runCommand = `docker run -d --name ${this.testId} -p ${this.testPort}:80 ${this.testId}`;
        const result = this.execCommand(runCommand, { silent: true });
        
        if (!result.success) {
            this.results.tests.container_start = false;
            this.results.errors.push(`Container start failed: ${result.error}`);
            throw new Error(`Container start failed: ${result.error}`);
        }
        
        // Wait for container to be ready
        this.log('Waiting for container to be ready...');
        await this.sleep(10000); // 10 seconds
        
        // Check if container is still running
        const psResult = this.execCommand(`docker ps --filter name=${this.testId} --format "{{.Names}}"`, { silent: true });
        if (!psResult.success || !psResult.output.includes(this.testId)) {
            // Get container logs for debugging
            const logsResult = this.execCommand(`docker logs ${this.testId}`, { silent: true });
            this.results.errors.push(`Container is not running. Logs: ${logsResult.output}`);
            throw new Error('Container failed to start or exited unexpectedly');
        }
        
        this.results.tests.container_start = true;
        this.log('Container started successfully', 'success');
    }

    /**
     * Test container health
     */
    async testContainerHealth() {
        this.log('Testing container health...', 'test');
        
        // Wait a bit more for health checks to initialize
        await this.sleep(5000);
        
        const healthResult = this.execCommand(`docker inspect --format='{{.State.Health.Status}}' ${this.testId}`, { silent: true });
        
        if (healthResult.success) {
            const healthStatus = healthResult.output.trim();
            this.results.containerInfo.healthStatus = healthStatus;
            
            if (healthStatus === 'healthy') {
                this.results.tests.container_health = true;
                this.log('Container health check passed', 'success');
            } else if (healthStatus === 'starting') {
                this.log('Container is still starting, waiting...', 'warning');
                await this.sleep(15000); // Wait 15 more seconds
                
                const retryResult = this.execCommand(`docker inspect --format='{{.State.Health.Status}}' ${this.testId}`, { silent: true });
                if (retryResult.success && retryResult.output.trim() === 'healthy') {
                    this.results.tests.container_health = true;
                    this.log('Container health check passed after retry', 'success');
                } else {
                    this.results.tests.container_health = false;
                    this.results.warnings.push(`Container health check failed: ${retryResult.output.trim()}`);
                }
            } else {
                this.results.tests.container_health = false;
                this.results.warnings.push(`Container health check failed: ${healthStatus}`);
            }
        } else {
            this.results.tests.container_health = false;
            this.results.warnings.push('Could not check container health status');
        }
    }

    /**
     * Test HTTP endpoints
     */
    async testHttpEndpoints() {
        this.log('Testing HTTP endpoints...', 'test');
        
        const endpoints = [
            { path: '/health', name: 'health_endpoint' },
            { path: '/', name: 'main_page' },
            { path: '/static/css/', name: 'static_assets', optional: true }
        ];
        
        for (const endpoint of endpoints) {
            try {
                const response = await this.makeHttpRequest(endpoint.path);
                
                if (response.statusCode >= 200 && response.statusCode < 400) {
                    this.results.tests[endpoint.name] = true;
                    this.log(`${endpoint.path} endpoint is accessible (${response.statusCode})`, 'success');
                } else {
                    this.results.tests[endpoint.name] = false;
                    if (endpoint.optional) {
                        this.results.warnings.push(`${endpoint.path} returned ${response.statusCode} (optional)`);
                    } else {
                        this.results.errors.push(`${endpoint.path} returned ${response.statusCode}`);
                    }
                }
            } catch (error) {
                this.results.tests[endpoint.name] = false;
                if (endpoint.optional) {
                    this.results.warnings.push(`${endpoint.path} failed: ${error.message} (optional)`);
                } else {
                    this.results.errors.push(`${endpoint.path} failed: ${error.message}`);
                }
            }
        }
    }

    /**
     * Test security headers
     */
    async testSecurityHeaders() {
        this.log('Testing security headers...', 'test');
        
        try {
            const response = await this.makeHttpRequest('/', { includeHeaders: true });
            
            const securityHeaders = [
                'X-Frame-Options',
                'X-Content-Type-Options',
                'X-XSS-Protection'
            ];
            
            let foundHeaders = 0;
            for (const header of securityHeaders) {
                if (response.headers[header.toLowerCase()]) {
                    foundHeaders++;
                }
            }
            
            this.results.tests.security_headers = foundHeaders > 0;
            this.results.metrics.security_headers_count = foundHeaders;
            
            if (foundHeaders > 0) {
                this.log(`Security headers present (${foundHeaders}/${securityHeaders.length})`, 'success');
            } else {
                this.results.warnings.push('No security headers found');
            }
            
        } catch (error) {
            this.results.tests.security_headers = false;
            this.results.warnings.push(`Security headers test failed: ${error.message}`);
        }
    }

    /**
     * Test compression
     */
    async testCompression() {
        this.log('Testing compression...', 'test');
        
        try {
            const response = await this.makeHttpRequest('/', { 
                includeHeaders: true,
                headers: { 'Accept-Encoding': 'gzip, deflate' }
            });
            
            const hasGzip = response.headers['content-encoding'] === 'gzip';
            this.results.tests.compression = hasGzip;
            
            if (hasGzip) {
                this.log('Gzip compression is working', 'success');
            } else {
                this.results.warnings.push('Gzip compression not detected');
            }
            
        } catch (error) {
            this.results.tests.compression = false;
            this.results.warnings.push(`Compression test failed: ${error.message}`);
        }
    }

    /**
     * Make HTTP request
     */
    makeHttpRequest(path, options = {}) {
        return new Promise((resolve, reject) => {
            const requestOptions = {
                hostname: 'localhost',
                port: this.testPort,
                path: path,
                method: 'GET',
                timeout: 10000,
                headers: options.headers || {}
            };

            const req = http.request(requestOptions, (res) => {
                let data = '';
                
                res.on('data', (chunk) => {
                    data += chunk;
                });
                
                res.on('end', () => {
                    resolve({
                        statusCode: res.statusCode,
                        headers: options.includeHeaders ? res.headers : undefined,
                        data: data
                    });
                });
            });

            req.on('error', (error) => {
                reject(error);
            });

            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            req.end();
        });
    }

    /**
     * Get container logs
     */
    getContainerLogs() {
        this.log('Collecting container logs...');
        
        const logsResult = this.execCommand(`docker logs ${this.testId}`, { silent: true });
        if (logsResult.success) {
            this.results.containerInfo.logs = logsResult.output;
        }
    }

    /**
     * Cleanup test resources
     */
    cleanupResources() {
        if (!this.cleanup) {
            this.log('Cleanup disabled, keeping test resources', 'warning');
            return;
        }
        
        this.log('Cleaning up test resources...', 'cleanup');
        
        // Stop and remove container
        this.execCommand(`docker stop ${this.testId} 2>/dev/null || true`, { silent: true });
        this.execCommand(`docker rm ${this.testId} 2>/dev/null || true`, { silent: true });
        
        // Remove image
        this.execCommand(`docker rmi ${this.testId} 2>/dev/null || true`, { silent: true });
        
        this.log('Cleanup completed', 'success');
    }

    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        this.log('Starting comprehensive Docker build testing...');
        
        try {
            // Pre-flight checks
            this.checkDockerAvailability();
            await this.checkPortAvailability();
            
            // Build and test
            await this.buildDockerImage();
            await this.startContainer();
            await this.testContainerHealth();
            await this.testHttpEndpoints();
            await this.testSecurityHeaders();
            await this.testCompression();
            
            // Collect logs
            this.getContainerLogs();
            
            // Calculate results
            const totalTime = Date.now() - this.startTime;
            this.results.metrics.total_test_time_ms = totalTime;
            
            const failedTests = Object.values(this.results.tests).filter(result => result === false).length;
            const totalTests = Object.keys(this.results.tests).length;
            
            this.results.metrics.tests_passed = totalTests - failedTests;
            this.results.metrics.tests_total = totalTests;
            this.results.metrics.success_rate = totalTests > 0 ? (totalTests - failedTests) / totalTests : 0;
            
            if (this.results.errors.length > 0) {
                this.results.status = 'failed';
            } else if (this.results.warnings.length > 0) {
                this.results.status = 'passed_with_warnings';
            } else {
                this.results.status = 'passed';
            }
            
            // Log summary
            this.log(`Testing completed in ${totalTime}ms`, this.results.status === 'passed' ? 'success' : 'warning');
            this.log(`Status: ${this.results.status}`);
            this.log(`Tests: ${this.results.metrics.tests_passed}/${this.results.metrics.tests_total} passed`);
            this.log(`Errors: ${this.results.errors.length}`);
            this.log(`Warnings: ${this.results.warnings.length}`);
            
        } catch (error) {
            this.results.status = 'failed';
            this.results.errors.push(`Test execution failed: ${error.message}`);
            this.log(`Test execution failed: ${error.message}`, 'error');
        } finally {
            this.cleanupResources();
        }
        
        return this.results;
    }

    /**
     * Generate test report
     */
    generateReport(outputPath = 'docker-build-test-report.json') {
        fs.writeFileSync(outputPath, JSON.stringify(this.results, null, 2));
        this.log(`Test report saved to ${outputPath}`);
    }
}

// CLI interface
if (require.main === module) {
    const tester = new DockerBuildTester();
    
    const args = process.argv.slice(2);
    const command = args[0] || 'test';
    
    // Parse arguments
    if (args.includes('--no-cleanup')) {
        tester.cleanup = false;
    }
    
    if (args.includes('--port')) {
        const portIndex = args.indexOf('--port');
        if (portIndex + 1 < args.length) {
            tester.testPort = parseInt(args[portIndex + 1]);
        }
    }
    
    if (command === 'help' || command === '--help') {
        console.log(`
Usage: node docker-build-test.js [command] [options]

Commands:
  test              Run all Docker build tests (default)
  help              Show this help message

Options:
  --no-cleanup      Keep test containers and images after testing
  --port <number>   Use specific port for testing (default: 3001)

Examples:
  node docker-build-test.js
  node docker-build-test.js test --no-cleanup
  node docker-build-test.js test --port 3002
        `);
        process.exit(0);
    }
    
    async function runTests() {
        try {
            const results = await tester.runAllTests();
            
            const outputFile = 'docker-build-test-report.json';
            tester.generateReport(outputFile);
            
            // Exit with appropriate code
            if (results.status === 'failed') {
                process.exit(1);
            } else if (results.status === 'passed_with_warnings') {
                process.exit(2);
            } else {
                process.exit(0);
            }
            
        } catch (error) {
            console.error('❌ Docker build testing failed:', error.message);
            process.exit(1);
        }
    }
    
    runTests();
}

module.exports = DockerBuildTester;