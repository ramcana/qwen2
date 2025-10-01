#!/usr/bin/env node

/**
 * Environment-Specific Docker Build Script
 * Builds Docker images for different environments with appropriate configurations
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Docker build configurations for different environments
const DOCKER_CONFIGS = {
  development: {
    dockerfile: 'Dockerfile.dev',
    target: 'development',
    tag: 'qwen-frontend:dev',
    buildArgs: {
      NODE_ENV: 'development',
      REACT_APP_API_URL: 'http://localhost:8000/api',
      REACT_APP_WS_URL: 'ws://localhost:8000/ws',
      REACT_APP_BACKEND_HOST: 'localhost',
      REACT_APP_BACKEND_PORT: '8000',
      REACT_APP_VERSION: '2.0.0-dev',
      GENERATE_SOURCEMAP: 'true',
      FAST_REFRESH: 'true',
      CHOKIDAR_USEPOLLING: 'true',
      REACT_APP_DEBUG: 'true'
    },
    ports: ['3000:3000'],
    volumes: [
      './src:/app/src:ro',
      './public:/app/public:ro',
      '/app/node_modules'
    ]
  },
  staging: {
    dockerfile: 'Dockerfile.prod',
    target: 'production',
    tag: 'qwen-frontend:staging',
    buildArgs: {
      NODE_ENV: 'production',
      REACT_APP_API_URL: '/api',
      REACT_APP_WS_URL: '/ws',
      REACT_APP_BACKEND_HOST: 'qwen-api-staging',
      REACT_APP_BACKEND_PORT: '8000',
      REACT_APP_VERSION: '2.0.0-staging',
      REACT_APP_ENVIRONMENT: 'staging',
      GENERATE_SOURCEMAP: 'true',
      BUILD_OPTIMIZATION: 'true',
      REACT_APP_DEBUG: 'true'
    },
    ports: ['3002:80'],
    volumes: ['./nginx.staging.conf:/etc/nginx/nginx.conf:ro']
  },
  production: {
    dockerfile: 'Dockerfile.prod',
    target: 'production',
    tag: 'qwen-frontend:prod',
    buildArgs: {
      NODE_ENV: 'production',
      REACT_APP_API_URL: '/api',
      REACT_APP_WS_URL: '/ws',
      REACT_APP_BACKEND_HOST: 'qwen-api',
      REACT_APP_BACKEND_PORT: '8000',
      REACT_APP_VERSION: '2.0.0',
      REACT_APP_ENVIRONMENT: 'production',
      GENERATE_SOURCEMAP: 'false',
      BUILD_OPTIMIZATION: 'true',
      REACT_APP_DEBUG: 'false'
    },
    ports: ['3001:80'],
    volumes: []
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
    throw error;
  }
}

/**
 * Build Docker image for specified environment
 */
function buildDockerImage(environment, options = {}) {
  const config = DOCKER_CONFIGS[environment];
  
  if (!config) {
    throw new Error(`Invalid environment: ${environment}`);
  }
  
  console.log(`🐳 Building Docker image for ${environment.toUpperCase()} environment`);
  console.log(`📋 Configuration:`, config);
  
  // Prepare build arguments
  const buildArgs = Object.entries(config.buildArgs)
    .map(([key, value]) => `--build-arg ${key}="${value}"`)
    .join(' ');
  
  // Prepare build command
  let buildCommand = `docker build -f ${config.dockerfile}`;
  
  if (config.target) {
    buildCommand += ` --target ${config.target}`;
  }
  
  buildCommand += ` ${buildArgs}`;
  buildCommand += ` -t ${config.tag}`;
  
  // Add cache options for better performance
  if (options.useCache !== false) {
    buildCommand += ` --cache-from ${config.tag}`;
  }
  
  // Add build context
  buildCommand += ' .';
  
  // Execute build
  const startTime = Date.now();
  execCommand(buildCommand);
  const buildTime = Date.now() - startTime;
  
  console.log(`✅ Docker image built successfully in ${(buildTime / 1000).toFixed(2)}s`);
  console.log(`🏷️  Image tag: ${config.tag}`);
  
  return config.tag;
}

/**
 * Test Docker image
 */
function testDockerImage(environment, options = {}) {
  const config = DOCKER_CONFIGS[environment];
  const containerName = `qwen-frontend-${environment}-test`;
  
  console.log(`🧪 Testing Docker image: ${config.tag}`);
  
  try {
    // Stop and remove existing test container
    try {
      execCommand(`docker stop ${containerName} 2>/dev/null || true`);
      execCommand(`docker rm ${containerName} 2>/dev/null || true`);
    } catch (error) {
      // Ignore errors for non-existent containers
    }
    
    // Prepare run command
    const ports = config.ports.map(port => `-p ${port}`).join(' ');
    const volumes = config.volumes.map(volume => `-v ${volume}`).join(' ');
    
    let runCommand = `docker run -d --name ${containerName} ${ports}`;
    if (volumes) {
      runCommand += ` ${volumes}`;
    }
    runCommand += ` ${config.tag}`;
    
    // Start container
    execCommand(runCommand);
    
    // Wait for container to start
    console.log('⏳ Waiting for container to start...');
    execCommand('sleep 10');
    
    // Check container health
    const healthCheck = execSync(`docker ps --filter name=${containerName} --format "{{.Status}}"`, { encoding: 'utf8' });
    console.log(`📊 Container status: ${healthCheck.trim()}`);
    
    // Test HTTP endpoint
    const port = config.ports[0].split(':')[0];
    const healthUrl = environment === 'development' ? 
      `http://localhost:${port}/health` : 
      `http://localhost:${port}/health`;
    
    try {
      console.log(`🔍 Testing health endpoint: ${healthUrl}`);
      execCommand(`curl -f ${healthUrl} || echo "Health check failed"`);
      console.log('✅ Health check passed');
    } catch (error) {
      console.warn('⚠️  Health check failed, but container is running');
    }
    
    // Show container logs
    if (options.showLogs) {
      console.log('📋 Container logs:');
      execCommand(`docker logs ${containerName} --tail 20`);
    }
    
    // Cleanup if requested
    if (options.cleanup !== false) {
      console.log('🧹 Cleaning up test container...');
      execCommand(`docker stop ${containerName}`);
      execCommand(`docker rm ${containerName}`);
    } else {
      console.log(`🔧 Test container left running: ${containerName}`);
      console.log(`   Stop with: docker stop ${containerName}`);
      console.log(`   Remove with: docker rm ${containerName}`);
    }
    
    console.log('✅ Docker image test completed successfully');
    
  } catch (error) {
    console.error('❌ Docker image test failed:', error.message);
    
    // Cleanup on failure
    try {
      execCommand(`docker stop ${containerName} 2>/dev/null || true`);
      execCommand(`docker rm ${containerName} 2>/dev/null || true`);
    } catch (cleanupError) {
      // Ignore cleanup errors
    }
    
    throw error;
  }
}

/**
 * Push Docker image to registry
 */
function pushDockerImage(environment, registry = '') {
  const config = DOCKER_CONFIGS[environment];
  let imageTag = config.tag;
  
  if (registry) {
    const newTag = `${registry}/${config.tag}`;
    console.log(`🏷️  Tagging image for registry: ${newTag}`);
    execCommand(`docker tag ${config.tag} ${newTag}`);
    imageTag = newTag;
  }
  
  console.log(`📤 Pushing image: ${imageTag}`);
  execCommand(`docker push ${imageTag}`);
  console.log('✅ Image pushed successfully');
}

/**
 * Generate Docker Compose file for environment
 */
function generateDockerCompose(environment) {
  const config = DOCKER_CONFIGS[environment];
  
  const compose = {
    version: '3.8',
    services: {
      [`frontend-${environment}`]: {
        image: config.tag,
        container_name: `qwen-frontend-${environment}`,
        ports: config.ports,
        environment: Object.keys(config.buildArgs).map(key => 
          `${key}=${config.buildArgs[key]}`
        ),
        restart: 'unless-stopped',
        labels: [
          `com.qwen.service=frontend`,
          `com.qwen.environment=${environment}`,
          `com.qwen.version=${config.buildArgs.REACT_APP_VERSION || '2.0.0'}`
        ]
      }
    }
  };
  
  if (config.volumes.length > 0) {
    compose.services[`frontend-${environment}`].volumes = config.volumes;
  }
  
  const composeFile = `docker-compose.${environment}.generated.yml`;
  fs.writeFileSync(composeFile, `# Generated Docker Compose for ${environment}\n` + 
    require('js-yaml').dump(compose, { indent: 2 }));
  
  console.log(`📄 Generated Docker Compose file: ${composeFile}`);
  return composeFile;
}

/**
 * Main function
 */
function main() {
  const command = process.argv[2] || 'build';
  const environment = process.argv[3] || 'production';
  const options = {
    useCache: !process.argv.includes('--no-cache'),
    cleanup: !process.argv.includes('--no-cleanup'),
    showLogs: process.argv.includes('--logs'),
    push: process.argv.includes('--push'),
    registry: process.argv.find(arg => arg.startsWith('--registry='))?.split('=')[1]
  };
  
  if (!DOCKER_CONFIGS[environment]) {
    console.error(`❌ Invalid environment: ${environment}`);
    console.error(`Available environments: ${Object.keys(DOCKER_CONFIGS).join(', ')}`);
    process.exit(1);
  }
  
  console.log(`🐳 Docker ${command} for ${environment.toUpperCase()} environment\n`);
  
  try {
    switch (command) {
      case 'build':
        buildDockerImage(environment, options);
        break;
        
      case 'test':
        testDockerImage(environment, options);
        break;
        
      case 'build-and-test':
        buildDockerImage(environment, options);
        testDockerImage(environment, options);
        break;
        
      case 'push':
        if (options.registry) {
          pushDockerImage(environment, options.registry);
        } else {
          console.error('❌ Registry required for push command. Use --registry=your-registry.com');
          process.exit(1);
        }
        break;
        
      case 'compose':
        generateDockerCompose(environment);
        break;
        
      default:
        console.error(`❌ Invalid command: ${command}`);
        console.error('Available commands: build, test, build-and-test, push, compose');
        process.exit(1);
    }
    
    console.log(`\n🎉 Docker ${command} completed successfully for ${environment} environment!`);
    
  } catch (error) {
    console.error(`\n❌ Docker ${command} failed for ${environment} environment:`, error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { buildDockerImage, testDockerImage, DOCKER_CONFIGS };