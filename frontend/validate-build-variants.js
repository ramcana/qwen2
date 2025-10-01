#!/usr/bin/env node

/**
 * Build Variants Validation Script
 * Validates that all build variants work correctly and produce expected outputs
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Test configurations for each variant
const testConfigs = {
  development: {
    expectedFiles: [
      'build-dev/index.html',
      'build-dev/static/js',
      'build-dev/static/css'
    ],
    expectedFeatures: {
      sourceMaps: true,
      minification: false,
      hotReload: true,
      debugging: true
    },
    buildCommand: 'npm run build:development',
    buildDir: 'build-dev'
  },
  staging: {
    expectedFiles: [
      'build/index.html',
      'build/static/js',
      'build/static/css',
      'build/build-info.json'
    ],
    expectedFeatures: {
      sourceMaps: true,
      minification: true,
      hotReload: false,
      debugging: true
    },
    buildCommand: 'npm run build:staging',
    buildDir: 'build'
  },
  production: {
    expectedFiles: [
      'build/index.html',
      'build/static/js',
      'build/static/css',
      'build/build-info.json'
    ],
    expectedFeatures: {
      sourceMaps: false,
      minification: true,
      hotReload: false,
      debugging: false
    },
    buildCommand: 'npm run build:production',
    buildDir: 'build'
  }
};

/**
 * Execute command with error handling
 */
function execCommand(command, options = {}) {
  try {
    console.log(`🔧 Executing: ${command}`);
    const result = execSync(command, { 
      stdio: options.silent ? 'pipe' : 'inherit', 
      encoding: 'utf8',
      ...options 
    });
    return result;
  } catch (error) {
    if (!options.allowFailure) {
      console.error(`❌ Command failed: ${command}`);
      console.error(error.message);
      throw error;
    }
    return null;
  }
}

/**
 * Check if file or directory exists
 */
function checkPath(filePath, type = 'file') {
  const exists = fs.existsSync(filePath);
  if (!exists) {
    return { exists: false, error: `${type} not found: ${filePath}` };
  }
  
  const stats = fs.statSync(filePath);
  const isCorrectType = type === 'file' ? stats.isFile() : stats.isDirectory();
  
  if (!isCorrectType) {
    return { exists: false, error: `Expected ${type} but found ${stats.isFile() ? 'file' : 'directory'}: ${filePath}` };
  }
  
  return { exists: true, size: stats.size };
}

/**
 * Validate build output structure
 */
function validateBuildStructure(variant, config) {
  console.log(`📁 Validating build structure for ${variant}...`);
  
  const errors = [];
  const warnings = [];
  
  // Check expected files and directories
  config.expectedFiles.forEach(expectedPath => {
    const fullPath = path.resolve(expectedPath);
    const isDirectory = !path.extname(expectedPath);
    const result = checkPath(fullPath, isDirectory ? 'directory' : 'file');
    
    if (!result.exists) {
      errors.push(result.error);
    } else {
      console.log(`  ✅ Found: ${expectedPath} (${result.size ? `${Math.round(result.size / 1024)}KB` : 'directory'})`);
    }
  });
  
  // Check for source maps
  const buildDir = config.buildDir;
  if (config.expectedFeatures.sourceMaps) {
    const sourceMaps = execCommand(`find ${buildDir} -name "*.map" 2>/dev/null || true`, { silent: true });
    if (!sourceMaps || sourceMaps.trim() === '') {
      warnings.push('Source maps expected but not found');
    } else {
      console.log(`  ✅ Source maps found`);
    }
  } else {
    const sourceMaps = execCommand(`find ${buildDir} -name "*.map" 2>/dev/null || true`, { silent: true });
    if (sourceMaps && sourceMaps.trim() !== '') {
      warnings.push('Source maps found but should be disabled in production');
    } else {
      console.log(`  ✅ Source maps correctly disabled`);
    }
  }
  
  // Check for minification
  if (config.expectedFeatures.minification) {
    const jsFiles = execCommand(`find ${buildDir}/static/js -name "*.js" 2>/dev/null | head -1 || true`, { silent: true });
    if (jsFiles && jsFiles.trim()) {
      const firstJsFile = jsFiles.trim().split('\n')[0];
      const content = fs.readFileSync(firstJsFile, 'utf8');
      const isMinified = content.length > 1000 && !content.includes('\n  '); // Simple minification check
      
      if (isMinified) {
        console.log(`  ✅ JavaScript appears to be minified`);
      } else {
        warnings.push('JavaScript files may not be properly minified');
      }
    }
  }
  
  return { errors, warnings };
}

/**
 * Validate environment configuration
 */
function validateEnvironmentConfig(variant) {
  console.log(`⚙️  Validating environment configuration for ${variant}...`);
  
  const errors = [];
  const warnings = [];
  
  // Check environment file
  const envFile = `.env.${variant}`;
  if (!fs.existsSync(envFile)) {
    errors.push(`Environment file not found: ${envFile}`);
    return { errors, warnings };
  }
  
  const envContent = fs.readFileSync(envFile, 'utf8');
  const envVars = {};
  
  envContent.split('\n').forEach(line => {
    line = line.trim();
    if (line && !line.startsWith('#')) {
      const [key, ...valueParts] = line.split('=');
      if (key && valueParts.length > 0) {
        envVars[key.trim()] = valueParts.join('=').trim();
      }
    }
  });
  
  // Validate required variables
  const requiredVars = [
    'NODE_ENV',
    'REACT_APP_API_URL',
    'REACT_APP_WS_URL',
    'REACT_APP_ENVIRONMENT'
  ];
  
  requiredVars.forEach(varName => {
    if (!envVars[varName]) {
      errors.push(`Missing required environment variable: ${varName}`);
    } else {
      console.log(`  ✅ ${varName}=${envVars[varName]}`);
    }
  });
  
  // Validate environment-specific settings
  if (variant === 'development') {
    if (envVars['REACT_APP_DEBUG'] !== 'true') {
      warnings.push('Debug mode should be enabled in development');
    }
    if (envVars['GENERATE_SOURCEMAP'] !== 'true') {
      warnings.push('Source maps should be enabled in development');
    }
  }
  
  if (variant === 'production') {
    if (envVars['REACT_APP_DEBUG'] === 'true') {
      warnings.push('Debug mode should be disabled in production');
    }
    if (envVars['GENERATE_SOURCEMAP'] === 'true') {
      warnings.push('Source maps should be disabled in production');
    }
  }
  
  return { errors, warnings };
}

/**
 * Validate Docker configuration
 */
function validateDockerConfig(variant) {
  console.log(`🐳 Validating Docker configuration for ${variant}...`);
  
  const errors = [];
  const warnings = [];
  
  // Check Dockerfile
  const dockerfile = variant === 'development' ? 'Dockerfile.dev' : 'Dockerfile.prod';
  if (!fs.existsSync(dockerfile)) {
    errors.push(`Dockerfile not found: ${dockerfile}`);
    return { errors, warnings };
  }
  
  console.log(`  ✅ Found Dockerfile: ${dockerfile}`);
  
  // Check docker-compose file
  const composeFile = `docker-compose.${variant === 'production' ? 'optimized' : variant}.yml`;
  if (!fs.existsSync(composeFile)) {
    warnings.push(`Docker Compose file not found: ${composeFile}`);
  } else {
    console.log(`  ✅ Found Docker Compose file: ${composeFile}`);
  }
  
  // Check nginx configuration
  const nginxConfig = `nginx.${variant === 'production' ? 'prod' : variant}.conf`;
  if (!fs.existsSync(nginxConfig)) {
    warnings.push(`Nginx configuration not found: ${nginxConfig}`);
  } else {
    console.log(`  ✅ Found nginx configuration: ${nginxConfig}`);
  }
  
  return { errors, warnings };
}

/**
 * Run build test for variant
 */
function testBuildVariant(variant) {
  console.log(`\n🏗️  Testing ${variant} build variant...`);
  
  const config = testConfigs[variant];
  if (!config) {
    throw new Error(`Unknown variant: ${variant}`);
  }
  
  let totalErrors = 0;
  let totalWarnings = 0;
  
  try {
    // Clean previous build
    console.log(`🧹 Cleaning previous build...`);
    execCommand(`rm -rf ${config.buildDir}`, { allowFailure: true });
    
    // Validate environment configuration
    const envValidation = validateEnvironmentConfig(variant);
    totalErrors += envValidation.errors.length;
    totalWarnings += envValidation.warnings.length;
    
    if (envValidation.errors.length > 0) {
      console.log(`  ❌ Environment validation errors:`);
      envValidation.errors.forEach(error => console.log(`     - ${error}`));
    }
    
    if (envValidation.warnings.length > 0) {
      console.log(`  ⚠️  Environment validation warnings:`);
      envValidation.warnings.forEach(warning => console.log(`     - ${warning}`));
    }
    
    // Validate Docker configuration
    const dockerValidation = validateDockerConfig(variant);
    totalErrors += dockerValidation.errors.length;
    totalWarnings += dockerValidation.warnings.length;
    
    if (dockerValidation.errors.length > 0) {
      console.log(`  ❌ Docker validation errors:`);
      dockerValidation.errors.forEach(error => console.log(`     - ${error}`));
    }
    
    if (dockerValidation.warnings.length > 0) {
      console.log(`  ⚠️  Docker validation warnings:`);
      dockerValidation.warnings.forEach(warning => console.log(`     - ${warning}`));
    }
    
    // Skip build if there are critical errors
    if (totalErrors > 0) {
      console.log(`❌ Skipping build due to ${totalErrors} error(s)`);
      return { success: false, errors: totalErrors, warnings: totalWarnings };
    }
    
    // Run build
    console.log(`🔨 Running build command: ${config.buildCommand}`);
    execCommand(config.buildCommand);
    
    // Validate build output
    const buildValidation = validateBuildStructure(variant, config);
    totalErrors += buildValidation.errors.length;
    totalWarnings += buildValidation.warnings.length;
    
    if (buildValidation.errors.length > 0) {
      console.log(`  ❌ Build validation errors:`);
      buildValidation.errors.forEach(error => console.log(`     - ${error}`));
    }
    
    if (buildValidation.warnings.length > 0) {
      console.log(`  ⚠️  Build validation warnings:`);
      buildValidation.warnings.forEach(warning => console.log(`     - ${warning}`));
    }
    
    // Generate build report
    const buildInfo = {
      variant,
      timestamp: new Date().toISOString(),
      buildDir: config.buildDir,
      success: totalErrors === 0,
      errors: totalErrors,
      warnings: totalWarnings,
      features: config.expectedFeatures
    };
    
    const reportPath = `build-validation-${variant}-${Date.now()}.json`;
    fs.writeFileSync(reportPath, JSON.stringify(buildInfo, null, 2));
    console.log(`📊 Build report saved: ${reportPath}`);
    
    if (totalErrors === 0) {
      console.log(`✅ ${variant} build variant validation passed`);
      return { success: true, errors: 0, warnings: totalWarnings };
    } else {
      console.log(`❌ ${variant} build variant validation failed with ${totalErrors} error(s)`);
      return { success: false, errors: totalErrors, warnings: totalWarnings };
    }
    
  } catch (error) {
    console.error(`❌ Build test failed for ${variant}:`, error.message);
    return { success: false, errors: totalErrors + 1, warnings: totalWarnings };
  }
}

/**
 * Main function
 */
function main() {
  const args = process.argv.slice(2);
  const variants = args.length > 0 ? args : Object.keys(testConfigs);
  
  console.log('🚀 Starting build variants validation...\n');
  console.log(`Testing variants: ${variants.join(', ')}\n`);
  
  const results = {};
  let totalErrors = 0;
  let totalWarnings = 0;
  
  variants.forEach(variant => {
    if (!testConfigs[variant]) {
      console.error(`❌ Unknown variant: ${variant}`);
      totalErrors++;
      return;
    }
    
    const result = testBuildVariant(variant);
    results[variant] = result;
    totalErrors += result.errors;
    totalWarnings += result.warnings;
  });
  
  // Summary
  console.log('\n📊 Validation Summary:');
  console.log('='.repeat(50));
  
  Object.entries(results).forEach(([variant, result]) => {
    const status = result.success ? '✅ PASS' : '❌ FAIL';
    console.log(`${status} ${variant}: ${result.errors} errors, ${result.warnings} warnings`);
  });
  
  console.log('='.repeat(50));
  console.log(`Total: ${totalErrors} errors, ${totalWarnings} warnings`);
  
  if (totalErrors > 0) {
    console.log('\n❌ Build variants validation failed');
    process.exit(1);
  } else if (totalWarnings > 0) {
    console.log('\n⚠️  Build variants validation completed with warnings');
    process.exit(0);
  } else {
    console.log('\n🎉 All build variants validation passed!');
    process.exit(0);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  testConfigs,
  testBuildVariant,
  validateBuildStructure,
  validateEnvironmentConfig,
  validateDockerConfig
};