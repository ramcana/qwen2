#!/usr/bin/env node
/**
 * Health Check Script for Frontend Container
 * Provides comprehensive health checking for the React frontend
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

class FrontendHealthChecker {
    constructor() {
        this.port = process.env.PORT || 80;
        this.host = process.env.HOST || 'localhost';
        this.timeout = 5000; // 5 seconds
        this.healthEndpoints = ['/health', '/'];
        this.staticPaths = ['/static/css', '/static/js'];
    }

    /**
     * Perform comprehensive health check
     */
    async performHealthCheck() {
        const results = {
            timestamp: new Date().toISOString(),
            service: 'qwen-frontend',
            status: 'healthy',
            checks: {},
            errors: [],
            warnings: []
        };

        try {
            // Check if nginx is responding
            await this.checkHttpEndpoint('/', results);
            
            // Check health endpoint specifically
            await this.checkHttpEndpoint('/health', results);
            
            // Check static file serving
            await this.checkStaticFiles(results);
            
            // Check file system access
            this.checkFileSystemAccess(results);
            
            // Check nginx configuration
            this.checkNginxConfig(results);
            
            // Determine overall status
            if (results.errors.length > 0) {
                results.status = 'unhealthy';
            } else if (results.warnings.length > 0) {
                results.status = 'degraded';
            }
            
            return results;
            
        } catch (error) {
            results.status = 'unhealthy';
            results.errors.push(`Health check failed: ${error.message}`);
            return results;
        }
    }

    /**
     * Check HTTP endpoint availability
     */
    async checkHttpEndpoint(path, results) {
        return new Promise((resolve, reject) => {
            const options = {
                hostname: this.host,
                port: this.port,
                path: path,
                method: 'GET',
                timeout: this.timeout
            };

            const req = http.request(options, (res) => {
                const checkName = `http_${path.replace('/', 'root').replace(/[^a-zA-Z0-9]/g, '_')}`;
                
                if (res.statusCode >= 200 && res.statusCode < 400) {
                    results.checks[checkName] = true;
                } else {
                    results.checks[checkName] = false;
                    results.warnings.push(`HTTP ${path} returned status ${res.statusCode}`);
                }
                
                resolve();
            });

            req.on('error', (error) => {
                const checkName = `http_${path.replace('/', 'root').replace(/[^a-zA-Z0-9]/g, '_')}`;
                results.checks[checkName] = false;
                results.errors.push(`HTTP ${path} failed: ${error.message}`);
                resolve(); // Don't reject, continue with other checks
            });

            req.on('timeout', () => {
                const checkName = `http_${path.replace('/', 'root').replace(/[^a-zA-Z0-9]/g, '_')}`;
                results.checks[checkName] = false;
                results.errors.push(`HTTP ${path} timed out`);
                req.destroy();
                resolve();
            });

            req.end();
        });
    }

    /**
     * Check static file serving
     */
    async checkStaticFiles(results) {
        const staticDir = '/usr/share/nginx/html/static';
        
        try {
            if (fs.existsSync(staticDir)) {
                const files = fs.readdirSync(staticDir);
                results.checks['static_files_exist'] = files.length > 0;
                
                if (files.length === 0) {
                    results.warnings.push('No static files found');
                }
            } else {
                results.checks['static_files_exist'] = false;
                results.warnings.push('Static directory does not exist');
            }
        } catch (error) {
            results.checks['static_files_exist'] = false;
            results.errors.push(`Static file check failed: ${error.message}`);
        }
    }

    /**
     * Check file system access
     */
    checkFileSystemAccess(results) {
        const criticalPaths = [
            '/usr/share/nginx/html',
            '/usr/share/nginx/html/index.html',
            '/etc/nginx/nginx.conf'
        ];

        for (const filePath of criticalPaths) {
            const checkName = `fs_access_${path.basename(filePath)}`;
            
            try {
                const exists = fs.existsSync(filePath);
                results.checks[checkName] = exists;
                
                if (!exists) {
                    results.errors.push(`Critical path missing: ${filePath}`);
                }
            } catch (error) {
                results.checks[checkName] = false;
                results.errors.push(`File system check failed for ${filePath}: ${error.message}`);
            }
        }
    }

    /**
     * Check nginx configuration
     */
    checkNginxConfig(results) {
        try {
            // Check if nginx config file exists and is readable
            const configPath = '/etc/nginx/nginx.conf';
            if (fs.existsSync(configPath)) {
                const config = fs.readFileSync(configPath, 'utf8');
                results.checks['nginx_config_readable'] = true;
                
                // Basic config validation
                const hasServerBlock = config.includes('server {');
                const hasListenDirective = config.includes('listen ');
                
                results.checks['nginx_config_valid'] = hasServerBlock && hasListenDirective;
                
                if (!hasServerBlock || !hasListenDirective) {
                    results.warnings.push('Nginx configuration may be incomplete');
                }
            } else {
                results.checks['nginx_config_readable'] = false;
                results.errors.push('Nginx configuration file not found');
            }
        } catch (error) {
            results.checks['nginx_config_readable'] = false;
            results.errors.push(`Nginx config check failed: ${error.message}`);
        }
    }

    /**
     * Simple health check for container orchestration
     */
    async simpleHealthCheck() {
        try {
            await this.checkHttpEndpoint('/', {checks: {}, errors: [], warnings: []});
            return true;
        } catch (error) {
            return false;
        }
    }
}

// CLI interface
if (require.main === module) {
    const checker = new FrontendHealthChecker();
    
    const command = process.argv[2] || 'full';
    
    if (command === 'simple') {
        // Simple health check for container health checks
        checker.simpleHealthCheck()
            .then(healthy => {
                if (healthy) {
                    console.log('OK');
                    process.exit(0);
                } else {
                    console.log('FAIL');
                    process.exit(1);
                }
            })
            .catch(error => {
                console.error('FAIL:', error.message);
                process.exit(1);
            });
    } else {
        // Full health check with detailed output
        checker.performHealthCheck()
            .then(results => {
                console.log(JSON.stringify(results, null, 2));
                process.exit(results.status === 'healthy' ? 0 : 1);
            })
            .catch(error => {
                console.error('Health check failed:', error.message);
                process.exit(1);
            });
    }
}

module.exports = FrontendHealthChecker;