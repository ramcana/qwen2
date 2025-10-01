#!/usr/bin/env node

/**
 * Build workaround for Node.js 22 compatibility issues
 * This script temporarily downgrades Node.js features to build the React app
 */

const { spawn } = require('child_process');
const path = require('path');

// Set environment variables for compatibility
process.env.NODE_OPTIONS = '--max-old-space-size=4096';
process.env.GENERATE_SOURCEMAP = 'false';
process.env.INLINE_RUNTIME_CHUNK = 'false';

// Force legacy OpenSSL provider for older webpack versions
if (process.version.startsWith('v18') || process.version.startsWith('v20') || process.version.startsWith('v22')) {
  process.env.NODE_OPTIONS += ' --openssl-legacy-provider';
}

console.log('Starting React build with compatibility settings...');
console.log('Node version:', process.version);
console.log('NODE_OPTIONS:', process.env.NODE_OPTIONS);

const buildProcess = spawn('npm', ['run', 'build'], {
  stdio: 'inherit',
  shell: true,
  cwd: __dirname
});

buildProcess.on('close', (code) => {
  if (code === 0) {
    console.log('Build completed successfully!');
  } else {
    console.error(`Build failed with exit code ${code}`);
    process.exit(code);
  }
});

buildProcess.on('error', (error) => {
  console.error('Build process error:', error);
  process.exit(1);
});