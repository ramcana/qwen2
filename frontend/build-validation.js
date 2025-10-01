#!/usr/bin/env node
/**
 * Enhanced Build Validation Script
 * Comprehensive validation for frontend build process
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

class BuildValidator {
    constructor() {
        this.results = {
            timestamp: new Date().toISOString(),
            service: 'qwen-frontend-build',
            status: 'unknown',
            validations: {},
            errors: [],
            warnings: [],
            metrics: {}
        };
        this.startTime = Date.now();
    }

    /**
     * Log validation step
     */
    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = {
            info: '🔍',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        }[type] || 'ℹ️';
        
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    /**
     * Validate package.json and dependencies
     */
    validateDependencies() {
        this.log('Validating dependencies...');
        
        try {
            // Check package.json exists
            if (!fs.existsSync('package.json')) {
                throw new Error('package.json not found');
            }
            
            const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
            this.results.validations.package_json_valid = true;
            
            // Check critical dependencies
            const criticalDeps = ['react', 'react-dom', 'react-scripts', 'typescript'];
            const missingDeps = [];
            
            for (const dep of criticalDeps) {
                if (!packageJson.dependencies[dep] && !packageJson.devDependencies[dep]) {
                    missingDeps.push(dep);
                }
            }
            
            if (missingDeps.length > 0) {
                throw new Error(`Missing critical dependencies: ${missingDeps.join(', ')}`);
            }
            
            this.results.validations.critical_dependencies_present = true;
            
            // Check for security vulnerabilities in dependencies
            try {
                execSync('npm audit --audit-level=high --json', { stdio: 'pipe' });
                this.results.validations.security_audit_passed = true;
            } catch (error) {
                this.results.validations.security_audit_passed = false;
                this.results.warnings.push('npm audit found security issues');
            }
            
            this.log('Dependencies validation passed', 'success');
            
        } catch (error) {
            this.results.validations.dependencies_valid = false;
            this.results.errors.push(`Dependencies validation failed: ${error.message}`);
            this.log(`Dependencies validation failed: ${error.message}`, 'error');
        }
    }

    /**
     * Validate source code structure
     */
    validateSourceStructure() {
        this.log('Validating source code structure...');
        
        try {
            const requiredFiles = [
                'src/index.tsx',
                'public/index.html',
                'tsconfig.json'
            ];
            
            const missingFiles = [];
            for (const file of requiredFiles) {
                if (!fs.existsSync(file)) {
                    missingFiles.push(file);
                }
            }
            
            if (missingFiles.length > 0) {
                throw new Error(`Missing required files: ${missingFiles.join(', ')}`);
            }
            
            this.results.validations.source_structure_valid = true;
            
            // Check TypeScript configuration
            const tsConfig = JSON.parse(fs.readFileSync('tsconfig.json', 'utf8'));
            if (!tsConfig.compilerOptions) {
                throw new Error('Invalid tsconfig.json: missing compilerOptions');
            }
            
            this.results.validations.typescript_config_valid = true;
            this.log('Source structure validation passed', 'success');
            
        } catch (error) {
            this.results.validations.source_structure_valid = false;
            this.results.errors.push(`Source structure validation failed: ${error.message}`);
            this.log(`Source structure validation failed: ${error.message}`, 'error');
        }
    }

    /**
     * Validate TypeScript compilation
     */
    async validateTypeScriptCompilation() {
        this.log('Validating TypeScript compilation...');
        
        try {
            execSync('npx tsc --noEmit --skipLibCheck', { stdio: 'pipe' });
            this.results.validations.typescript_compilation = true;
            this.log('TypeScript compilation validation passed', 'success');
            
        } catch (error) {
            this.results.validations.typescript_compilation = false;
            this.results.errors.push('TypeScript compilation failed');
            this.log('TypeScript compilation failed', 'error');
        }
    }

    /**
     * Validate build process
     */
    async validateBuildProcess() {
        this.log('Validating build process...');
        
        const buildStartTime = Date.now();
        
        try {
            // Clean existing build
            if (fs.existsSync('build')) {
                fs.rmSync('build', { recursive: true, force: true });
            }
            
            // Run build
            execSync('npm run build:production', { stdio: 'pipe' });
            
            const buildTime = Date.now() - buildStartTime;
            this.results.metrics.build_time_ms = buildTime;
            
            // Validate build output
            if (!fs.existsSync('build')) {
                throw new Error('Build directory not created');
            }
            
            if (!fs.existsSync('build/index.html')) {
                throw new Error('Build output missing index.html');
            }
            
            // Check build size
            const buildStats = this.getBuildStats();
            this.results.metrics.build_size_bytes = buildStats.totalSize;
            this.results.metrics.build_files_count = buildStats.fileCount;
            
            // Validate HTML content
            const indexHtml = fs.readFileSync('build/index.html', 'utf8');
            if (!indexHtml.includes('<div id="root">')) {
                throw new Error('Build output HTML missing root element');
            }
            
            this.results.validations.build_process = true;
            this.log(`Build process validation passed (${buildTime}ms, ${buildStats.totalSize} bytes)`, 'success');
            
        } catch (error) {
            this.results.validations.build_process = false;
            this.results.errors.push(`Build process validation failed: ${error.message}`);
            this.log(`Build process validation failed: ${error.message}`, 'error');
        }
    }

    /**
     * Get build statistics
     */
    getBuildStats() {
        const buildDir = 'build';
        let totalSize = 0;
        let fileCount = 0;
        
        const calculateSize = (dir) => {
            const files = fs.readdirSync(dir);
            
            for (const file of files) {
                const filePath = path.join(dir, file);
                const stat = fs.statSync(filePath);
                
                if (stat.isDirectory()) {
                    calculateSize(filePath);
                } else {
                    totalSize += stat.size;
                    fileCount++;
                }
            }
        };
        
        if (fs.existsSync(buildDir)) {
            calculateSize(buildDir);
        }
        
        return { totalSize, fileCount };
    }

    /**
     * Validate Docker configuration
     */
    validateDockerConfiguration() {
        this.log('Validating Docker configuration...');
        
        try {
            const dockerFiles = [
                'Dockerfile.prod',
                'nginx.prod.conf',
                'nginx-security.conf'
            ];
            
            const missingFiles = [];
            for (const file of dockerFiles) {
                if (!fs.existsSync(file)) {
                    missingFiles.push(file);
                }
            }
            
            if (missingFiles.length > 0) {
                throw new Error(`Missing Docker files: ${missingFiles.join(', ')}`);
            }
            
            // Validate Dockerfile syntax (basic check)
            const dockerfile = fs.readFileSync('Dockerfile.prod', 'utf8');
            if (!dockerfile.includes('FROM ') || !dockerfile.includes('COPY ')) {
                throw new Error('Invalid Dockerfile structure');
            }
            
            this.results.validations.docker_configuration = true;
            this.log('Docker configuration validation passed', 'success');
            
        } catch (error) {
            this.results.validations.docker_configuration = false;
            this.results.errors.push(`Docker configuration validation failed: ${error.message}`);
            this.log(`Docker configuration validation failed: ${error.message}`, 'error');
        }
    }

    /**
     * Validate environment configuration
     */
    validateEnvironmentConfiguration() {
        this.log('Validating environment configuration...');
        
        try {
            // Check for environment files
            const envFiles = ['.env', '.env.local'];
            let hasEnvFile = false;
            
            for (const file of envFiles) {
                if (fs.existsSync(file)) {
                    hasEnvFile = true;
                    break;
                }
            }
            
            if (!hasEnvFile) {
                this.results.warnings.push('No environment configuration files found');
            }
            
            // Validate required environment variables for build
            const requiredEnvVars = ['REACT_APP_API_URL'];
            const missingEnvVars = [];
            
            for (const envVar of requiredEnvVars) {
                if (!process.env[envVar]) {
                    missingEnvVars.push(envVar);
                }
            }
            
            if (missingEnvVars.length > 0) {
                this.results.warnings.push(`Missing environment variables: ${missingEnvVars.join(', ')}`);
            }
            
            this.results.validations.environment_configuration = true;
            this.log('Environment configuration validation passed', 'success');
            
        } catch (error) {
            this.results.validations.environment_configuration = false;
            this.results.errors.push(`Environment configuration validation failed: ${error.message}`);
            this.log(`Environment configuration validation failed: ${error.message}`, 'error');
        }
    }

    /**
     * Run all validations
     */
    async runAllValidations() {
        this.log('Starting comprehensive build validation...');
        
        // Run validations in sequence
        this.validateDependencies();
        this.validateSourceStructure();
        await this.validateTypeScriptCompilation();
        this.validateDockerConfiguration();
        this.validateEnvironmentConfiguration();
        
        // Run build validation last (most time-consuming)
        await this.validateBuildProcess();
        
        // Calculate overall status
        const totalTime = Date.now() - this.startTime;
        this.results.metrics.total_validation_time_ms = totalTime;
        
        const hasErrors = this.results.errors.length > 0;
        const hasWarnings = this.results.warnings.length > 0;
        
        if (hasErrors) {
            this.results.status = 'failed';
        } else if (hasWarnings) {
            this.results.status = 'passed_with_warnings';
        } else {
            this.results.status = 'passed';
        }
        
        // Log summary
        this.log(`Validation completed in ${totalTime}ms`, this.results.status === 'passed' ? 'success' : 'warning');
        this.log(`Status: ${this.results.status}`);
        this.log(`Errors: ${this.results.errors.length}`);
        this.log(`Warnings: ${this.results.warnings.length}`);
        
        return this.results;
    }

    /**
     * Generate validation report
     */
    generateReport(outputPath = 'build-validation-report.json') {
        fs.writeFileSync(outputPath, JSON.stringify(this.results, null, 2));
        this.log(`Validation report saved to ${outputPath}`);
    }
}

// CLI interface
if (require.main === module) {
    const validator = new BuildValidator();
    
    const command = process.argv[2] || 'full';
    const outputFile = process.argv[3] || 'build-validation-report.json';
    
    if (command === 'help' || command === '--help') {
        console.log(`
Usage: node build-validation.js [command] [output-file]

Commands:
  full              Run all validations (default)
  dependencies      Validate dependencies only
  structure         Validate source structure only
  typescript        Validate TypeScript compilation only
  build             Validate build process only
  docker            Validate Docker configuration only
  environment       Validate environment configuration only
  help              Show this help message

Output:
  output-file       Path to save validation report (default: build-validation-report.json)
        `);
        process.exit(0);
    }
    
    async function runValidation() {
        try {
            let results;
            
            switch (command) {
                case 'dependencies':
                    validator.validateDependencies();
                    results = validator.results;
                    break;
                case 'structure':
                    validator.validateSourceStructure();
                    results = validator.results;
                    break;
                case 'typescript':
                    await validator.validateTypeScriptCompilation();
                    results = validator.results;
                    break;
                case 'build':
                    await validator.validateBuildProcess();
                    results = validator.results;
                    break;
                case 'docker':
                    validator.validateDockerConfiguration();
                    results = validator.results;
                    break;
                case 'environment':
                    validator.validateEnvironmentConfiguration();
                    results = validator.results;
                    break;
                default:
                    results = await validator.runAllValidations();
            }
            
            validator.generateReport(outputFile);
            
            // Exit with appropriate code
            if (results.status === 'failed') {
                process.exit(1);
            } else if (results.status === 'passed_with_warnings') {
                process.exit(2);
            } else {
                process.exit(0);
            }
            
        } catch (error) {
            console.error('❌ Validation failed:', error.message);
            process.exit(1);
        }
    }
    
    runValidation();
}

module.exports = BuildValidator;