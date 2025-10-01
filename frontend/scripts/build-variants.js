#!/usr/bin/env node

/**
 * Build Variants Manager
 * Manages different build configurations for development, staging, and production
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Build variant configurations
const buildVariants = {
  development: {
    dockerfile: 'Dockerfile.dev',
    envFile: '.env.development',
    buildArgs: {
      NODE_ENV: 'development',
      REACT_APP_API_URL: 'http://localhost:8000/api',
      REACT_APP_WS_URL: 'ws://localhost:8000/ws',
      REACT_APP_BACKEND_HOST: 'localhost',
      REACT_APP_BACKEND_PORT: '8000',
      GENERATE_SOURCEMAP: 'true',
      FAST_REFRESH: 'true',
      REACT_APP_DEBUG: 'true'
    },
    nginxConfig: 'nginx.dev.conf',
    optimizations: {
      minification: false,
      compression: false,
      sourceMaps: true,
      hotReload: true
    }
  },
  staging: {
    dockerfile: 'Dockerfile.prod',
    envFile: '.env.staging',
    buildArgs: {
      NODE_ENV: 'production',
      REACT_APP_API_URL: '/api',
      REACT_APP_WS_URL: '/ws',
      REACT_APP_BACKEND_HOST: 'qwen-api-staging',
      REACT_APP_BACKEND_PORT: '8000',
      REACT_APP_VERSION: '2.0.0-staging',
      GENERATE_SOURCEMAP: 'true',
      BUILD_OPTIMIZATION: 'true'
    },
    nginxConfig: 'nginx.staging.conf',
    optimizations: {
      minification: true,
      compression: true,
      sourceMaps: true,
      hotReload: false
    }
  },
  production: {
    dockerfile: 'Dockerfile.prod',
    envFile: '.env.production',
    buildArgs: {
      NODE_ENV: 'production',
      REACT_APP_API_URL: '/api',
      REACT_APP_WS_URL: '/ws',
      REACT_APP_BACKEND_HOST: 'qwen-api',
      REACT_APP_BACKEND_PORT: '8000',
      REACT_APP_VERSION: '2.0.0',
      GENERATE_SOURCEMAP: 'false',
      BUILD_OPTIMIZATION: 'true'
    },
    nginxConfig: 'nginx.prod.conf',
    optimizations: {
      minification: true,
      compression: true,
      sourceMaps: false,
      hotReload: false
    }
  }
};

/**
 * Execute command with error handling
 */
function execCommand(command, options = {}) {
  try {
    console.log(`🔧 Executing: ${command}`);
    const result = execSync(command, { 
      stdio: 'inherit', 
      encoding: 'utf8',
      ...options 
    });
    return result;
  } catch (error) {
    console.error(`❌ Command failed: ${command}`);
    console.error(error.message);
    throw error;
  }
}

/**
 * Validate build variant configuration
 */
function validateVariant(variant) {
  const config = buildVariants[variant];
  if (!config) {
    throw new Error(`Unknown build variant: ${variant}`);
  }

  // Check if required files exist
  const requiredFiles = [config.dockerfile, config.envFile];
  const missingFiles = requiredFiles.filter(file => !fs.existsSync(file));
  
  if (missingFiles.length > 0) {
    throw new Error(`Missing required files for ${variant}: ${missingFiles.join(', ')}`);
  }

  return config;
}

/**
 * Build Docker image for specific variant
 */
function buildDockerImage(variant, options = {}) {
  console.log(`🏗️  Building Docker image for ${variant} variant...`);
  
  const config = validateVariant(variant);
  const timestamp = new Date().toISOString();
  
  // Prepare build arguments
  const buildArgs = Object.entries(config.buildArgs)
    .map(([key, value]) => `--build-arg ${key}="${value}"`)
    .join(' ');

  // Prepare image tags
  const baseTag = options.tag || `qwen-frontend:${variant}`;
  const timestampTag = `${baseTag}-${timestamp.replace(/[:.]/g, '-')}`;
  
  // Build command
  const buildCommand = [
    'docker build',
    `-f ${config.dockerfile}`,
    buildArgs,
    `--target ${variant === 'development' ? 'development' : 'production'}`,
    `-t ${baseTag}`,
    `-t ${timestampTag}`,
    options.cache ? '--cache-from qwen-frontend:cache' : '',
    options.noCache ? '--no-cache' : '',
    '.'
  ].filter(Boolean).join(' ');

  execCommand(buildCommand);

  console.log(`✅ Successfully built ${variant} image: ${baseTag}`);
  return baseTag;
}

/**
 * Build local variant (non-Docker)
 */
function buildLocal(variant, options = {}) {
  console.log(`🏗️  Building local ${variant} variant...`);
  
  const config = validateVariant(variant);
  
  // Set environment variables
  const envVars = Object.entries(config.buildArgs)
    .map(([key, value]) => `${key}="${value}"`)
    .join(' ');

  // Build command
  const buildCommand = `${envVars} npm run build:${variant}`;
  
  execCommand(buildCommand);
  
  console.log(`✅ Successfully built ${variant} locally`);
}

/**
 * Run variant-specific tests
 */
function testVariant(variant, options = {}) {
  console.log(`🧪 Testing ${variant} variant...`);
  
  const config = validateVariant(variant);
  
  // Test commands based on variant
  const testCommands = {
    development: [
      'npm run lint',
      'npm run test:dev -- --ci --watchAll=false --coverage'
    ],
    staging: [
      'npm run lint',
      'npm run test:staging',
      'npm run validate:build'
    ],
    production: [
      'npm run lint',
      'npm run test:ci',
      'npm run validate:build:full'
    ]
  };

  const commands = testCommands[variant] || testCommands.production;
  
  commands.forEach(command => {
    execCommand(command);
  });
  
  console.log(`✅ All tests passed for ${variant} variant`);
}

/**
 * Deploy variant
 */
function deployVariant(variant, options = {}) {
  console.log(`🚀 Deploying ${variant} variant...`);
  
  const config = validateVariant(variant);
  
  // Deployment commands based on variant
  const deployCommands = {
    development: [
      `docker-compose -f docker-compose.dev.yml up -d --build`
    ],
    staging: [
      `docker-compose -f docker-compose.staging.yml up -d --build`
    ],
    production: [
      `docker-compose -f docker-compose.prod.yml up -d --build`
    ]
  };

  const commands = deployCommands[variant];
  if (!commands) {
    throw new Error(`No deployment configuration for ${variant}`);
  }
  
  commands.forEach(command => {
    execCommand(command);
  });
  
  console.log(`✅ Successfully deployed ${variant} variant`);
}

/**
 * Generate build report
 */
function generateBuildReport(variant, buildResult) {
  const config = buildVariants[variant];
  const timestamp = new Date().toISOString();
  
  const report = {
    variant,
    timestamp,
    configuration: config,
    buildResult,
    optimizations: config.optimizations,
    environment: config.buildArgs
  };

  const reportPath = `build-report-${variant}-${timestamp.replace(/[:.]/g, '-')}.json`;
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  
  console.log(`📊 Build report generated: ${reportPath}`);
  return reportPath;
}

/**
 * Main CLI function
 */
function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  const variant = args[1];
  const options = {};

  // Parse options
  args.slice(2).forEach(arg => {
    if (arg.startsWith('--')) {
      const [key, value] = arg.substring(2).split('=');
      options[key] = value || true;
    }
  });

  if (!command || !variant) {
    console.log(`
Usage: node build-variants.js <command> <variant> [options]

Commands:
  build     Build the specified variant
  test      Test the specified variant
  deploy    Deploy the specified variant
  validate  Validate variant configuration

Variants:
  development   Development build with hot reload
  staging       Staging build with debugging
  production    Production build optimized

Options:
  --docker      Use Docker build (default for staging/production)
  --local       Use local build (default for development)
  --no-cache    Disable build cache
  --tag=<name>  Custom Docker tag

Examples:
  node build-variants.js build development --local
  node build-variants.js build staging --docker
  node build-variants.js build production --docker --no-cache
  node build-variants.js test staging
  node build-variants.js deploy production
    `);
    process.exit(1);
  }

  try {
    switch (command) {
      case 'build':
        if (options.docker || (variant !== 'development' && !options.local)) {
          const imageTag = buildDockerImage(variant, options);
          generateBuildReport(variant, { type: 'docker', tag: imageTag });
        } else {
          buildLocal(variant, options);
          generateBuildReport(variant, { type: 'local' });
        }
        break;
        
      case 'test':
        testVariant(variant, options);
        break;
        
      case 'deploy':
        deployVariant(variant, options);
        break;
        
      case 'validate':
        validateVariant(variant);
        console.log(`✅ ${variant} variant configuration is valid`);
        break;
        
      default:
        throw new Error(`Unknown command: ${command}`);
    }
    
    console.log(`\n🎉 ${command} completed successfully for ${variant} variant`);
    
  } catch (error) {
    console.error(`\n❌ ${command} failed for ${variant} variant:`);
    console.error(error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  buildVariants,
  buildDockerImage,
  buildLocal,
  testVariant,
  deployVariant,
  validateVariant,
  generateBuildReport
};