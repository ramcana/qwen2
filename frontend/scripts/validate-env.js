#!/usr/bin/env node

/**
 * Environment Configuration Validator
 * Validates environment-specific configurations and build settings
 */

const fs = require('fs');
const path = require('path');

// Environment configurations to validate
const environments = ['development', 'staging', 'production'];

// Required environment variables for each environment
const requiredVars = {
  development: [
    'NODE_ENV',
    'REACT_APP_API_URL',
    'REACT_APP_WS_URL',
    'REACT_APP_BACKEND_HOST',
    'REACT_APP_BACKEND_PORT',
    'REACT_APP_DEBUG',
    'GENERATE_SOURCEMAP',
    'FAST_REFRESH'
  ],
  staging: [
    'NODE_ENV',
    'REACT_APP_API_URL',
    'REACT_APP_WS_URL',
    'REACT_APP_BACKEND_HOST',
    'REACT_APP_BACKEND_PORT',
    'REACT_APP_ENVIRONMENT',
    'BUILD_OPTIMIZATION'
  ],
  production: [
    'NODE_ENV',
    'REACT_APP_API_URL',
    'REACT_APP_WS_URL',
    'REACT_APP_BACKEND_HOST',
    'REACT_APP_BACKEND_PORT',
    'REACT_APP_ENVIRONMENT',
    'BUILD_OPTIMIZATION'
  ]
};

// Environment-specific validation rules
const validationRules = {
  development: {
    'NODE_ENV': 'development',
    'REACT_APP_DEBUG': 'true',
    'GENERATE_SOURCEMAP': 'true',
    'FAST_REFRESH': 'true'
  },
  staging: {
    'NODE_ENV': 'production',
    'REACT_APP_ENVIRONMENT': 'staging',
    'BUILD_OPTIMIZATION': 'true'
  },
  production: {
    'NODE_ENV': 'production',
    'REACT_APP_ENVIRONMENT': 'production',
    'REACT_APP_DEBUG': 'false',
    'GENERATE_SOURCEMAP': 'false',
    'BUILD_OPTIMIZATION': 'true'
  }
};

/**
 * Parse environment file
 */
function parseEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  const content = fs.readFileSync(filePath, 'utf8');
  const env = {};

  content.split('\n').forEach(line => {
    line = line.trim();
    if (line && !line.startsWith('#')) {
      const [key, ...valueParts] = line.split('=');
      if (key && valueParts.length > 0) {
        env[key.trim()] = valueParts.join('=').trim();
      }
    }
  });

  return env;
}

/**
 * Validate environment configuration
 */
function validateEnvironment(env, envVars, rules) {
  const errors = [];
  const warnings = [];

  // Check required variables
  requiredVars[env].forEach(varName => {
    if (!envVars[varName]) {
      errors.push(`Missing required variable: ${varName}`);
    }
  });

  // Check validation rules
  Object.entries(rules[env] || {}).forEach(([key, expectedValue]) => {
    if (envVars[key] && envVars[key] !== expectedValue) {
      warnings.push(`${key} is "${envVars[key]}", expected "${expectedValue}"`);
    }
  });

  // Environment-specific validations
  if (env === 'development') {
    if (envVars['REACT_APP_API_URL'] && !envVars['REACT_APP_API_URL'].includes('localhost')) {
      warnings.push('Development should typically use localhost for API URL');
    }
  }

  if (env === 'production') {
    if (envVars['REACT_APP_DEBUG'] === 'true') {
      warnings.push('Debug mode should be disabled in production');
    }
    if (envVars['GENERATE_SOURCEMAP'] === 'true') {
      warnings.push('Source maps should be disabled in production for security');
    }
  }

  return { errors, warnings };
}

/**
 * Validate Docker configurations
 */
function validateDockerConfigs() {
  const dockerFiles = [
    'Dockerfile.dev',
    'Dockerfile.prod',
    'docker-compose.dev.yml',
    'docker-compose.staging.yml',
    'docker-compose.optimized.yml'
  ];

  const missing = dockerFiles.filter(file => !fs.existsSync(file));
  return missing;
}

/**
 * Validate nginx configurations
 */
function validateNginxConfigs() {
  const nginxFiles = [
    'nginx.dev.conf',
    'nginx.staging.conf',
    'nginx.prod.conf'
  ];

  const missing = nginxFiles.filter(file => !fs.existsSync(file));
  return missing;
}

/**
 * Main validation function
 */
function main() {
  console.log('🔍 Validating environment configurations...\n');

  let hasErrors = false;
  let hasWarnings = false;

  // Validate each environment
  environments.forEach(env => {
    console.log(`📋 Validating ${env} environment:`);
    
    const envFile = `.env.${env}`;
    const envVars = parseEnvFile(envFile);

    if (!envVars) {
      console.log(`  ❌ Environment file ${envFile} not found`);
      hasErrors = true;
      return;
    }

    const { errors, warnings } = validateEnvironment(env, envVars, validationRules);

    if (errors.length > 0) {
      hasErrors = true;
      console.log(`  ❌ Errors:`);
      errors.forEach(error => console.log(`     - ${error}`));
    }

    if (warnings.length > 0) {
      hasWarnings = true;
      console.log(`  ⚠️  Warnings:`);
      warnings.forEach(warning => console.log(`     - ${warning}`));
    }

    if (errors.length === 0 && warnings.length === 0) {
      console.log(`  ✅ Configuration valid`);
    }

    console.log('');
  });

  // Validate Docker configurations
  console.log('🐳 Validating Docker configurations:');
  const missingDockerFiles = validateDockerConfigs();
  if (missingDockerFiles.length > 0) {
    hasErrors = true;
    console.log(`  ❌ Missing Docker files:`);
    missingDockerFiles.forEach(file => console.log(`     - ${file}`));
  } else {
    console.log(`  ✅ All Docker files present`);
  }
  console.log('');

  // Validate nginx configurations
  console.log('🌐 Validating nginx configurations:');
  const missingNginxFiles = validateNginxConfigs();
  if (missingNginxFiles.length > 0) {
    hasWarnings = true;
    console.log(`  ⚠️  Missing nginx files:`);
    missingNginxFiles.forEach(file => console.log(`     - ${file}`));
  } else {
    console.log(`  ✅ All nginx files present`);
  }
  console.log('');

  // Validate package.json scripts
  console.log('📦 Validating package.json scripts:');
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const requiredScripts = [
    'start:dev',
    'build:development',
    'build:staging',
    'build:production',
    'docker:build:dev',
    'docker:build:staging',
    'docker:build:prod'
  ];

  const missingScripts = requiredScripts.filter(script => !packageJson.scripts[script]);
  if (missingScripts.length > 0) {
    hasWarnings = true;
    console.log(`  ⚠️  Missing scripts:`);
    missingScripts.forEach(script => console.log(`     - ${script}`));
  } else {
    console.log(`  ✅ All required scripts present`);
  }

  // Summary
  console.log('\n📊 Validation Summary:');
  if (hasErrors) {
    console.log('  ❌ Validation failed with errors');
    process.exit(1);
  } else if (hasWarnings) {
    console.log('  ⚠️  Validation completed with warnings');
    process.exit(0);
  } else {
    console.log('  ✅ All validations passed');
    process.exit(0);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  parseEnvFile,
  validateEnvironment,
  validateDockerConfigs,
  validateNginxConfigs
};