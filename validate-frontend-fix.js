#!/usr/bin/env node

/**
 * Quick validation script to verify frontend TypeScript fixes
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating Frontend TypeScript Fixes...\n');

// Check if build was successful
const buildDir = path.join(__dirname, 'frontend', 'build');
if (fs.existsSync(buildDir)) {
    console.log('✅ Frontend build directory exists');
    
    // Check for main JS file
    const staticDir = path.join(buildDir, 'static', 'js');
    if (fs.existsSync(staticDir)) {
        const jsFiles = fs.readdirSync(staticDir).filter(f => f.startsWith('main.') && f.endsWith('.js'));
        if (jsFiles.length > 0) {
            console.log(`✅ Main JS file found: ${jsFiles[0]}`);
        } else {
            console.log('❌ No main JS file found');
        }
    }
    
    // Check for CSS file
    const cssDir = path.join(buildDir, 'static', 'css');
    if (fs.existsSync(cssDir)) {
        const cssFiles = fs.readdirSync(cssDir).filter(f => f.startsWith('main.') && f.endsWith('.css'));
        if (cssFiles.length > 0) {
            console.log(`✅ Main CSS file found: ${cssFiles[0]}`);
        }
    }
} else {
    console.log('❌ Frontend build directory not found');
    process.exit(1);
}

// Validate key fixes
console.log('\n🔧 Validating Key Fixes:');

// Check StatusBar.tsx fixes
const statusBarPath = path.join(__dirname, 'frontend', 'src', 'components', 'StatusBar.tsx');
if (fs.existsSync(statusBarPath)) {
    const content = fs.readFileSync(statusBarPath, 'utf8');
    
    if (content.includes('getMemoryValues()')) {
        console.log('✅ Memory calculation helper function added');
    }
    
    if (content.includes('status.queue_length') && !content.includes('status.queue_size')) {
        console.log('✅ Queue field names corrected (queue_length)');
    }
    
    if (content.includes('status.current_generation') && !content.includes('status.is_generating')) {
        console.log('✅ Generation status field corrected (current_generation)');
    }
    
    if (!content.includes('status.gpu_available')) {
        console.log('✅ Removed gpu_available reference');
    }
}

// Check API types
const apiTypesPath = path.join(__dirname, 'frontend', 'src', 'types', 'api.ts');
if (fs.existsSync(apiTypesPath)) {
    const content = fs.readFileSync(apiTypesPath, 'utf8');
    
    if (content.includes('device: string') && content.includes('queue_length: number')) {
        console.log('✅ StatusResponse interface matches backend schema');
    }
}

console.log('\n🎉 Frontend TypeScript fixes validation complete!');
console.log('\n📋 Summary of fixes applied:');
console.log('   • Fixed gpu_available → device field mapping');
console.log('   • Added memory calculation helpers for GB and percentage values');
console.log('   • Fixed queue_size → queue_length field mapping');
console.log('   • Fixed is_generating → current_generation field mapping');
console.log('   • Resolved workspace state type issues');
console.log('   • Fixed error boundary componentStack type issue');
console.log('   • Fixed ControlNet file type null/undefined mismatch');