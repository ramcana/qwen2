#!/usr/bin/env node

/**
 * Environment-Specific Build Script
 * Handles building the React application for different environments
 * with appropriate optimizations and configurations
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Build configurations for different environments
const BUILD_CONFIGS = {
  development: {
    sourceMaps: true,
    optimization: false,
    minification: false,
    compression: false,
    buildPath: 'build-dev',
    nodeOptions: '--max-old-space-size=2048'
  },
  staging: {
    sourceMaps: true,
    optimization: true,
    minification: true,
    compression: true,
    buildPath: 'build-staging',
    nodeOptions: '--max-old-space-size=4096'
  },
  production: {
    sourceMaps: false,
    optimization: true,
    minification: true,
    compression: true,
    buildPath: 'build',
    nodeOptions: '--max-old-space-size=4096 --optimize-for-size'
  }
};

/**
 * Execute command with proper error handling
 */
function execCommand(command, options = {}) {
  try {
    console.log(`🔧 Executing: ${command}`);
    const result = execSync(command, {
      stdio: 'inherit',
      ...options
    });
    return result;
  } catch (error) {
    console.error(`❌ Command failed: ${command}`);
    console.error(error.message);
    process.exit(1);
  }
}

/**
 * Clean previous build artifacts
 */
function cleanBuild(environment) {
  const config = BUILD_CONFIGS[environment];
  const buildPath = path.join(__dirname, '..', config.buildPath);
  
  console.log(`🧹 Cleaning previous build: ${config.buildPath}`);
  
  if (fs.existsSync(buildPath)) {
    execCommand(`rm -rf ${buildPath}`);
  }
  
  // Clean cache
  const cachePath = path.join(__dirname, '..', 'node_modules', '.cache');
  if (fs.existsSync(cachePath)) {
    console.log('🧹 Cleaning build cache');
    execCommand(`rm -rf ${cachePath}`);
  }
}

/**
 * Validate environment before building
 */
function validateEnvironment(environment) {
  console.log(`🔍 Validating ${environment} environment...`);
  
  const envFile = path.join(__dirname, '..', `.env.${environment}`);
  if (!fs.existsSync(envFile)) {
    console.error(`❌ Environment file not found: .env.${environment}`);
    process.exit(1);
  }
  
  // Run environment validation script
  try {
    execCommand(`node ${path.join(__dirname, 'validate-env.js')}`);
  } catch (error) {
    console.error('❌ Environment validation failed');
    process.exit(1);
  }
}

/**
 * Build the application for the specified environment
 */
function buildApplication(environment) {
  const config = BUILD_CONFIGS[environment];
  
  console.log(`🚀 Building application for ${environment.toUpperCase()} environment`);
  console.log(`📊 Configuration:`, config);
  
  // Set Node.js options for build
  process.env.NODE_OPTIONS = config.nodeOptions;
  
  // Set build path
  process.env.BUILD_PATH = config.buildPath;
  
  // Build command based on environment
  let buildCommand;
  if (environment === 'development') {
    buildCommand = 'npm run build:development';
  } else if (environment === 'staging') {
    buildCommand = 'npm run build:staging';
  } else {
    buildCommand = 'npm run build:production';
  }
  
  // Execute build
  const startTime = Date.now();
  execCommand(buildCommand);
  const buildTime = Date.now() - startTime;
  
  console.log(`✅ Build completed in ${(buildTime / 1000).toFixed(2)}s`);
  
  return config.buildPath;
}

/**
 * Post-build optimizations
 */
function postBuildOptimizations(environment, buildPath) {
  const config = BUILD_CONFIGS[environment];
  const fullBuildPath = path.join(__dirname, '..', buildPath);
  
  console.log(`🔧 Applying post-build optimizations for ${environment}...`);
  
  // Generate build info
  const buildInfo = {
    environment,
    buildTime: new Date().toISOString(),
    version: process.env.REACT_APP_VERSION || '2.0.0',
    nodeEnv: process.env.NODE_ENV,
    sourceMaps: config.sourceMaps,
    optimization: config.optimization,
    buildPath: buildPath
  };
  
  fs.writeFileSync(
    path.join(fullBuildPath, 'build-info.json'),
    JSON.stringify(buildInfo, null, 2)
  );
  
  // Compression for staging and production
  if (config.compression && (environment === 'staging' || environment === 'production')) {
    console.log('🗜️  Compressing static assets...');
    
    // Gzip compression
    try {
      execCommand(`find ${fullBuildPath} -type f \\( -name "*.js" -o -name "*.css" -o -name "*.html" -o -name "*.json" -o -name "*.svg" \\) -exec gzip -k -9 {} \\;`);
      console.log('✅ Gzip compression completed');
    } catch (error) {
      console.warn('⚠️  Gzip compression failed:', error.message);
    }
    
    // Brotli compression (if available)
    try {
      execCommand(`find ${fullBuildPath} -type f \\( -name "*.js" -o -name "*.css" -o -name "*.html" -o -name "*.json" -o -name "*.svg" \\) -exec brotli -k -q 11 {} \\; 2>/dev/null || true`);
      console.log('✅ Brotli compression completed');
    } catch (error) {
      console.warn('⚠️  Brotli compression skipped (not available)');
    }
  }
  
  // Build analysis for production
  if (environment === 'production') {
    console.log('📊 Generating build analysis...');
    
    // Calculate build size
    try {
      const sizeOutput = execSync(`du -sh ${fullBuildPath}`, { encoding: 'utf8' });
      const buildSize = sizeOutput.split('\t')[0];
      console.log(`📦 Total build size: ${buildSize}`);
      
      // List largest files
      console.log('📋 Largest files:');
      execCommand(`find ${fullBuildPath} -type f -exec ls -lh {} \\; | sort -k5 -hr | head -10`);
    } catch (error) {
      console.warn('⚠️  Build analysis failed:', error.message);
    }
  }
}

/**
 * Validate build output
 */
function validateBuild(buildPath) {
  const fullBuildPath = path.join(__dirname, '..', buildPath);
  
  console.log(`🔍 Validating build output: ${buildPath}`);
  
  // Check required files
  const requiredFiles = ['index.html', 'static'];
  const missingFiles = requiredFiles.filter(file => 
    !fs.existsSync(path.join(fullBuildPath, file))
  );
  
  if (missingFiles.length > 0) {
    console.error(`❌ Missing required files: ${missingFiles.join(', ')}`);
    process.exit(1);
  }
  
  // Check index.html content
  const indexPath = path.join(fullBuildPath, 'index.html');
  const indexContent = fs.readFileSync(indexPath, 'utf8');
  
  if (!indexContent.includes('<div id="root">')) {
    console.error('❌ Invalid index.html: missing root div');
    process.exit(1);
  }
  
  console.log('✅ Build validation passed');
}

/**
 * Main build function
 */
function main() {
  const environment = process.argv[2] || 'production';
  
  if (!BUILD_CONFIGS[environment]) {
    console.error(`❌ Invalid environment: ${environment}`);
    console.error(`Available environments: ${Object.keys(BUILD_CONFIGS).join(', ')}`);
    process.exit(1);
  }
  
  console.log(`🏗️  Starting ${environment.toUpperCase()} build process...\n`);
  
  try {
    // Step 1: Validate environment
    validateEnvironment(environment);
    
    // Step 2: Clean previous builds
    cleanBuild(environment);
    
    // Step 3: Build application
    const buildPath = buildApplication(environment);
    
    // Step 4: Post-build optimizations
    postBuildOptimizations(environment, buildPath);
    
    // Step 5: Validate build
    validateBuild(buildPath);
    
    console.log(`\n🎉 ${environment.toUpperCase()} build completed successfully!`);
    console.log(`📁 Build output: ${buildPath}`);
    
  } catch (error) {
    console.error(`\n❌ Build failed for ${environment} environment:`, error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { buildApplication, BUILD_CONFIGS };