#!/usr/bin/env node
/**
 * Automated Build Testing Suite
 * Comprehensive automated testing for frontend build process
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Import our validation and testing modules
const BuildValidator = require('./build-validation.js');
const DockerBuildTester = require('./docker-build-test.js');

class AutomatedBuildTestSuite {
    constructor() {
        this.results = {
            timestamp: new Date().toISOString(),
            suite: 'automated-build-test',
            status: 'unknown',
            phases: {},
            summary: {
                total_phases: 0,
                passed_phases: 0,
                failed_phases: 0,
                warnings_count: 0,
                errors_count: 0
            },
            reports: {},
            metrics: {}
        };
        this.startTime = Date.now();
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
            phase: '📋',
            summary: '📊'
        }[type] || 'ℹ️';
        
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    /**
     * Run build validation phase
     */
    async runBuildValidation() {
        this.log('Running build validation phase...', 'phase');
        
        const phaseStartTime = Date.now();
        
        try {
            const validator = new BuildValidator();
            const validationResults = await validator.runAllValidations();
            
            const phaseTime = Date.now() - phaseStartTime;
            
            this.results.phases.build_validation = {
                status: validationResults.status,
                duration_ms: phaseTime,
                errors: validationResults.errors,
                warnings: validationResults.warnings,
                validations: validationResults.validations,
                metrics: validationResults.metrics
            };
            
            // Save detailed report
            const reportPath = 'build-validation-report.json';
            validator.generateReport(reportPath);
            this.results.reports.build_validation = reportPath;
            
            if (validationResults.status === 'failed') {
                this.log(`Build validation failed (${phaseTime}ms)`, 'error');
                return false;
            } else {
                this.log(`Build validation passed (${phaseTime}ms)`, 'success');
                return true;
            }
            
        } catch (error) {
            const phaseTime = Date.now() - phaseStartTime;
            this.results.phases.build_validation = {
                status: 'failed',
                duration_ms: phaseTime,
                errors: [error.message],
                warnings: []
            };
            
            this.log(`Build validation phase failed: ${error.message}`, 'error');
            return false;
        }
    }

    /**
     * Run Docker build testing phase
     */
    async runDockerBuildTesting() {
        this.log('Running Docker build testing phase...', 'phase');
        
        const phaseStartTime = Date.now();
        
        try {
            const tester = new DockerBuildTester();
            const testResults = await tester.runAllTests();
            
            const phaseTime = Date.now() - phaseStartTime;
            
            this.results.phases.docker_build_testing = {
                status: testResults.status,
                duration_ms: phaseTime,
                errors: testResults.errors,
                warnings: testResults.warnings,
                tests: testResults.tests,
                metrics: testResults.metrics,
                containerInfo: testResults.containerInfo
            };
            
            // Save detailed report
            const reportPath = 'docker-build-test-report.json';
            tester.generateReport(reportPath);
            this.results.reports.docker_build_testing = reportPath;
            
            if (testResults.status === 'failed') {
                this.log(`Docker build testing failed (${phaseTime}ms)`, 'error');
                return false;
            } else {
                this.log(`Docker build testing passed (${phaseTime}ms)`, 'success');
                return true;
            }
            
        } catch (error) {
            const phaseTime = Date.now() - phaseStartTime;
            this.results.phases.docker_build_testing = {
                status: 'failed',
                duration_ms: phaseTime,
                errors: [error.message],
                warnings: []
            };
            
            this.log(`Docker build testing phase failed: ${error.message}`, 'error');
            return false;
        }
    }

    /**
     * Run dependency compatibility check
     */
    async runDependencyCompatibilityCheck() {
        this.log('Running dependency compatibility check...', 'phase');
        
        const phaseStartTime = Date.now();
        
        try {
            // Check for outdated dependencies
            let outdatedResult;
            try {
                const outdatedOutput = execSync('npm outdated --json', { encoding: 'utf8', stdio: 'pipe' });
                outdatedResult = JSON.parse(outdatedOutput || '{}');
            } catch (error) {
                // npm outdated returns non-zero exit code when outdated packages are found
                if (error.stdout) {
                    try {
                        outdatedResult = JSON.parse(error.stdout);
                    } catch (parseError) {
                        outdatedResult = {};
                    }
                } else {
                    outdatedResult = {};
                }
            }
            
            // Check for security vulnerabilities
            let auditResult;
            try {
                execSync('npm audit --json', { encoding: 'utf8', stdio: 'pipe' });
                auditResult = { vulnerabilities: {} };
            } catch (error) {
                if (error.stdout) {
                    try {
                        auditResult = JSON.parse(error.stdout);
                    } catch (parseError) {
                        auditResult = { vulnerabilities: {} };
                    }
                } else {
                    auditResult = { vulnerabilities: {} };
                }
            }
            
            const phaseTime = Date.now() - phaseStartTime;
            
            const outdatedCount = Object.keys(outdatedResult).length;
            const vulnerabilityCount = Object.keys(auditResult.vulnerabilities || {}).length;
            
            const warnings = [];
            if (outdatedCount > 0) {
                warnings.push(`${outdatedCount} outdated dependencies found`);
            }
            if (vulnerabilityCount > 0) {
                warnings.push(`${vulnerabilityCount} security vulnerabilities found`);
            }
            
            this.results.phases.dependency_compatibility = {
                status: vulnerabilityCount > 0 ? 'failed' : (outdatedCount > 0 ? 'passed_with_warnings' : 'passed'),
                duration_ms: phaseTime,
                errors: vulnerabilityCount > 0 ? [`${vulnerabilityCount} security vulnerabilities found`] : [],
                warnings: warnings,
                metrics: {
                    outdated_dependencies: outdatedCount,
                    security_vulnerabilities: vulnerabilityCount
                },
                details: {
                    outdated: outdatedResult,
                    audit: auditResult
                }
            };
            
            if (vulnerabilityCount > 0) {
                this.log(`Dependency compatibility check failed: ${vulnerabilityCount} vulnerabilities`, 'error');
                return false;
            } else {
                this.log(`Dependency compatibility check passed (${phaseTime}ms)`, 'success');
                return true;
            }
            
        } catch (error) {
            const phaseTime = Date.now() - phaseStartTime;
            this.results.phases.dependency_compatibility = {
                status: 'failed',
                duration_ms: phaseTime,
                errors: [error.message],
                warnings: []
            };
            
            this.log(`Dependency compatibility check failed: ${error.message}`, 'error');
            return false;
        }
    }

    /**
     * Run integration test
     */
    async runIntegrationTest() {
        this.log('Running integration test...', 'phase');
        
        const phaseStartTime = Date.now();
        
        try {
            // Test that build artifacts can be served
            if (!fs.existsSync('build')) {
                throw new Error('Build directory not found - run build validation first');
            }
            
            // Check build artifacts
            const buildFiles = fs.readdirSync('build');
            if (!buildFiles.includes('index.html')) {
                throw new Error('index.html not found in build directory');
            }
            
            // Test static file serving (if static directory exists)
            let staticFilesCount = 0;
            if (fs.existsSync('build/static')) {
                const countFiles = (dir) => {
                    const files = fs.readdirSync(dir);
                    let count = 0;
                    for (const file of files) {
                        const filePath = path.join(dir, file);
                        if (fs.statSync(filePath).isDirectory()) {
                            count += countFiles(filePath);
                        } else {
                            count++;
                        }
                    }
                    return count;
                };
                staticFilesCount = countFiles('build/static');
            }
            
            const phaseTime = Date.now() - phaseStartTime;
            
            this.results.phases.integration_test = {
                status: 'passed',
                duration_ms: phaseTime,
                errors: [],
                warnings: [],
                metrics: {
                    build_files_count: buildFiles.length,
                    static_files_count: staticFilesCount
                }
            };
            
            this.log(`Integration test passed (${phaseTime}ms)`, 'success');
            return true;
            
        } catch (error) {
            const phaseTime = Date.now() - phaseStartTime;
            this.results.phases.integration_test = {
                status: 'failed',
                duration_ms: phaseTime,
                errors: [error.message],
                warnings: []
            };
            
            this.log(`Integration test failed: ${error.message}`, 'error');
            return false;
        }
    }

    /**
     * Calculate summary statistics
     */
    calculateSummary() {
        const phases = Object.values(this.results.phases);
        
        this.results.summary.total_phases = phases.length;
        this.results.summary.passed_phases = phases.filter(p => p.status === 'passed' || p.status === 'passed_with_warnings').length;
        this.results.summary.failed_phases = phases.filter(p => p.status === 'failed').length;
        
        this.results.summary.errors_count = phases.reduce((sum, p) => sum + (p.errors?.length || 0), 0);
        this.results.summary.warnings_count = phases.reduce((sum, p) => sum + (p.warnings?.length || 0), 0);
        
        this.results.summary.success_rate = this.results.summary.total_phases > 0 
            ? this.results.summary.passed_phases / this.results.summary.total_phases 
            : 0;
        
        const totalTime = Date.now() - this.startTime;
        this.results.metrics.total_test_time_ms = totalTime;
        
        // Determine overall status
        if (this.results.summary.failed_phases > 0) {
            this.results.status = 'failed';
        } else if (this.results.summary.warnings_count > 0) {
            this.results.status = 'passed_with_warnings';
        } else {
            this.results.status = 'passed';
        }
    }

    /**
     * Run all test phases
     */
    async runAllTests() {
        this.log('Starting automated build test suite...', 'phase');
        
        const phases = [
            { name: 'Dependency Compatibility Check', fn: () => this.runDependencyCompatibilityCheck() },
            { name: 'Build Validation', fn: () => this.runBuildValidation() },
            { name: 'Integration Test', fn: () => this.runIntegrationTest() },
            { name: 'Docker Build Testing', fn: () => this.runDockerBuildTesting() }
        ];
        
        let continueTests = true;
        
        for (const phase of phases) {
            if (!continueTests) {
                this.log(`Skipping ${phase.name} due to previous failures`, 'warning');
                continue;
            }
            
            try {
                const success = await phase.fn();
                if (!success) {
                    continueTests = false;
                }
            } catch (error) {
                this.log(`Phase ${phase.name} threw an error: ${error.message}`, 'error');
                continueTests = false;
            }
        }
        
        this.calculateSummary();
        
        // Log summary
        this.log('Test suite completed', 'summary');
        this.log(`Overall Status: ${this.results.status}`, this.results.status === 'passed' ? 'success' : 'warning');
        this.log(`Phases: ${this.results.summary.passed_phases}/${this.results.summary.total_phases} passed`);
        this.log(`Success Rate: ${(this.results.summary.success_rate * 100).toFixed(1)}%`);
        this.log(`Total Time: ${this.results.metrics.total_test_time_ms}ms`);
        this.log(`Errors: ${this.results.summary.errors_count}`);
        this.log(`Warnings: ${this.results.summary.warnings_count}`);
        
        return this.results;
    }

    /**
     * Generate comprehensive test report
     */
    generateReport(outputPath = 'automated-build-test-report.json') {
        fs.writeFileSync(outputPath, JSON.stringify(this.results, null, 2));
        this.log(`Comprehensive test report saved to ${outputPath}`);
        
        // Generate summary report
        const summaryPath = 'automated-build-test-summary.json';
        const summary = {
            timestamp: this.results.timestamp,
            status: this.results.status,
            summary: this.results.summary,
            metrics: this.results.metrics,
            phase_statuses: Object.fromEntries(
                Object.entries(this.results.phases).map(([name, phase]) => [name, phase.status])
            )
        };
        
        fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
        this.log(`Summary report saved to ${summaryPath}`);
    }
}

// CLI interface
if (require.main === module) {
    const testSuite = new AutomatedBuildTestSuite();
    
    const command = process.argv[2] || 'test';
    
    if (command === 'help' || command === '--help') {
        console.log(`
Usage: node automated-build-test.js [command]

Commands:
  test              Run complete automated build test suite (default)
  help              Show this help message

This script runs a comprehensive automated test suite that includes:
- Dependency compatibility checking
- Build validation
- Integration testing
- Docker build testing

The script generates detailed reports and exits with appropriate status codes:
- 0: All tests passed
- 1: Tests failed
- 2: Tests passed with warnings
        `);
        process.exit(0);
    }
    
    async function runTestSuite() {
        try {
            const results = await testSuite.runAllTests();
            
            testSuite.generateReport();
            
            // Exit with appropriate code
            if (results.status === 'failed') {
                process.exit(1);
            } else if (results.status === 'passed_with_warnings') {
                process.exit(2);
            } else {
                process.exit(0);
            }
            
        } catch (error) {
            console.error('❌ Automated build test suite failed:', error.message);
            process.exit(1);
        }
    }
    
    runTestSuite();
}

module.exports = AutomatedBuildTestSuite;