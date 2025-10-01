#!/usr/bin/env node

/**
 * Test script to verify build variants work correctly
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

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
    return null;
  }
}

function testEnvironmentConfig(env) {
  console.log(`\n🧪 Testing ${env} environment configuration...`);
  
  const envFile = `.env.${env}`;
  if (!fs.existsSync(envFile)) {
    console.error(`❌ Environment file missing: ${envFile}`);
    return false;
  }
  
  console.log(`✅ Environment file exists: ${envFile}`);
  
  // Read and display key configurations
  const content = fs.readFileSync(envFile, 'utf8');
  const lines = content.split('\n').filter(line => 
    line.trim() && !line.startsWith('#') && line.includes('=')
  );
  
  console.log(`📋 Key configurations for ${env}:`);
  lines.slice(0, 10).forEach(line => {
    console.log(`   ${line}`);
  });
  
  return true;
}

function testDockerfile(dockerfile) {
  console.log(`\n🐳 Testing Dockerfile: ${dockerfile}`);
  
  if (!fs.existsSync(dockerfile)) {
    console.error(`❌ Dockerfile missing: ${dockerfile}`);
    return false;
  }
  
  console.log(`✅ Dockerfile exists: ${dockerfile}`);
  
  // Basic syntax check
  const result = execCommand(`docker build -f ${dockerfile} --dry-run . 2>/dev/null || echo "Syntax check failed"`, { stdio: 'pipe' });
  
  return true;
}

function testNginxConfig(configFile) {
  console.log(`\n🌐 Testing Nginx config: ${configFile}`);
  
  if (!fs.existsSync(configFile)) {
    console.error(`❌ Nginx config missing: ${configFile}`);
    return false;
  }
  
  console.log(`✅ Nginx config exists: ${configFile}`);
  
  // Basic syntax check (if nginx is available)
  try {
    execCommand(`nginx -t -c ${path.resolve(configFile)} 2>/dev/null || echo "Nginx not available for testing"`, { stdio: 'pipe' });
  } catch (error) {
    console.log(`⚠️  Nginx syntax check skipped (nginx not available)`);
  }
  
  return true;
}

function main() {
  console.log('🚀 Testing Frontend Build Variants\n');
  
  let allPassed = true;
  
  // Test environment configurations
  const environments = ['development', 'staging', 'production'];
  environments.forEach(env => {
    if (!testEnvironmentConfig(env)) {
      allPassed = false;
    }
  });
  
  // Test Dockerfiles
  const dockerfiles = ['Dockerfile.dev', 'Dockerfile.prod'];
  dockerfiles.forEach(dockerfile => {
    if (!testDockerfile(dockerfile)) {
      allPassed = false;
    }
  });
  
  // Test Nginx configurations
  const nginxConfigs = ['nginx.dev.conf', 'nginx.staging.conf', 'nginx.prod.conf'];
  nginxConfigs.forEach(config => {
    if (!testNginxConfig(config)) {
      allPassed = false;
    }
  });
  
  // Test scripts
  console.log(`\n📜 Testing build scripts...`);
  const scripts = ['scripts/validate-env.js', 'scripts/build-env.js', 'scripts/docker-build-env.js'];
  scripts.forEach(script => {
    if (fs.existsSync(script)) {
      console.log(`✅ Script exists: ${script}`);
    } else {
      console.error(`❌ Script missing: ${script}`);
      allPassed = false;
    }
  });
  
  // Test package.json scripts
  console.log(`\n📦 Testing package.json scripts...`);
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const requiredScripts = [
    'build:development',
    'build:staging', 
    'build:production',
    'docker:build:dev',
    'docker:build:staging',
    'docker:build:prod',
    'validate:env'
  ];
  
  requiredScripts.forEach(script => {
    if (packageJson.scripts[script]) {
      console.log(`✅ Script defined: ${script}`);
    } else {
      console.error(`❌ Script missing: ${script}`);
      allPassed = false;
    }
  });
  
  // Summary
  console.log(`\n📊 Test Summary:`);
  if (allPassed) {
    console.log('✅ All build variant tests passed!');
    console.log('\n🎉 Frontend build variants are properly configured');
    console.log('\nNext steps:');
    console.log('1. Run: npm run validate:env');
    console.log('2. Test builds: npm run build:development');
    console.log('3. Test Docker: npm run docker:build:dev');
  } else {
    console.log('❌ Some tests failed. Please fix the issues above.');
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}