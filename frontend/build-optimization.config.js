/**
 * Build Optimization Configuration
 * Environment-specific build optimizations for different deployment targets
 */

const path = require('path');

// Base optimization settings
const baseOptimizations = {
  // Code splitting
  splitChunks: {
    chunks: 'all',
    cacheGroups: {
      vendor: {
        test: /[\\/]node_modules[\\/]/,
        name: 'vendors',
        chunks: 'all',
        priority: 10
      },
      common: {
        name: 'common',
        minChunks: 2,
        chunks: 'all',
        priority: 5,
        reuseExistingChunk: true
      }
    }
  },
  
  // Asset optimization
  assets: {
    maxSize: 250000, // 250KB
    maxAssetSize: 500000, // 500KB
    maxEntrypointSize: 500000 // 500KB
  },
  
  // Bundle analysis
  analyze: false,
  
  // Source maps
  sourceMaps: true,
  
  // Minification
  minify: true,
  
  // Compression
  compression: {
    gzip: true,
    brotli: true
  }
};

// Environment-specific configurations
const environmentConfigs = {
  development: {
    ...baseOptimizations,
    
    // Development-specific settings
    sourceMaps: true,
    minify: false,
    analyze: false,
    
    // Faster builds for development
    splitChunks: {
      chunks: 'async', // Only async chunks for faster builds
      cacheGroups: {
        default: false,
        vendors: false
      }
    },
    
    // Relaxed asset limits for development
    assets: {
      maxSize: 1000000, // 1MB
      maxAssetSize: 2000000, // 2MB
      maxEntrypointSize: 2000000 // 2MB
    },
    
    // No compression in development
    compression: {
      gzip: false,
      brotli: false
    },
    
    // Development-specific webpack optimizations
    webpack: {
      mode: 'development',
      devtool: 'eval-source-map',
      optimization: {
        minimize: false,
        splitChunks: false,
        runtimeChunk: false
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src')
        }
      }
    }
  },
  
  staging: {
    ...baseOptimizations,
    
    // Staging-specific settings (balanced)
    sourceMaps: true, // Keep source maps for debugging
    minify: true,
    analyze: true, // Enable bundle analysis for optimization insights
    
    // Moderate code splitting for staging
    splitChunks: {
      chunks: 'all',
      minSize: 20000,
      maxSize: 200000,
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
          priority: 10,
          maxSize: 300000
        },
        common: {
          name: 'common',
          minChunks: 2,
          chunks: 'all',
          priority: 5,
          reuseExistingChunk: true,
          maxSize: 200000
        },
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
          name: 'react',
          chunks: 'all',
          priority: 20
        }
      }
    },
    
    // Staging asset limits
    assets: {
      maxSize: 300000, // 300KB
      maxAssetSize: 600000, // 600KB
      maxEntrypointSize: 600000 // 600KB
    },
    
    // Enable compression for staging
    compression: {
      gzip: true,
      brotli: true
    },
    
    // Staging webpack optimizations
    webpack: {
      mode: 'production',
      devtool: 'source-map',
      optimization: {
        minimize: true,
        splitChunks: {
          chunks: 'all',
          cacheGroups: {
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              chunks: 'all'
            }
          }
        },
        runtimeChunk: 'single'
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src')
        }
      }
    }
  },
  
  production: {
    ...baseOptimizations,
    
    // Production-specific settings (maximum optimization)
    sourceMaps: false, // Disable source maps for security
    minify: true,
    analyze: false, // Disable analysis in production builds
    
    // Aggressive code splitting for production
    splitChunks: {
      chunks: 'all',
      minSize: 10000,
      maxSize: 150000,
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
          priority: 10,
          maxSize: 200000
        },
        common: {
          name: 'common',
          minChunks: 2,
          chunks: 'all',
          priority: 5,
          reuseExistingChunk: true,
          maxSize: 150000
        },
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
          name: 'react',
          chunks: 'all',
          priority: 20,
          maxSize: 200000
        },
        ui: {
          test: /[\\/]node_modules[\\/](@tanstack|lucide-react|clsx)[\\/]/,
          name: 'ui',
          chunks: 'all',
          priority: 15,
          maxSize: 150000
        }
      }
    },
    
    // Strict asset limits for production
    assets: {
      maxSize: 200000, // 200KB
      maxAssetSize: 400000, // 400KB
      maxEntrypointSize: 400000 // 400KB
    },
    
    // Maximum compression for production
    compression: {
      gzip: true,
      brotli: true,
      level: 9 // Maximum compression level
    },
    
    // Production webpack optimizations
    webpack: {
      mode: 'production',
      devtool: false,
      optimization: {
        minimize: true,
        splitChunks: {
          chunks: 'all',
          minSize: 10000,
          maxSize: 150000,
          cacheGroups: {
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendors',
              chunks: 'all',
              priority: 10
            },
            common: {
              name: 'common',
              minChunks: 2,
              chunks: 'all',
              priority: 5
            }
          }
        },
        runtimeChunk: 'single',
        moduleIds: 'deterministic',
        chunkIds: 'deterministic'
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, 'src')
        }
      },
      performance: {
        hints: 'error',
        maxAssetSize: 400000,
        maxEntrypointSize: 400000
      }
    }
  }
};

/**
 * Get configuration for specific environment
 */
function getConfig(environment = 'production') {
  const config = environmentConfigs[environment];
  if (!config) {
    throw new Error(`Unknown environment: ${environment}. Available: ${Object.keys(environmentConfigs).join(', ')}`);
  }
  return config;
}

/**
 * Get webpack configuration for environment
 */
function getWebpackConfig(environment = 'production') {
  const config = getConfig(environment);
  return config.webpack || {};
}

/**
 * Get optimization settings for environment
 */
function getOptimizationSettings(environment = 'production') {
  const config = getConfig(environment);
  return {
    splitChunks: config.splitChunks,
    assets: config.assets,
    compression: config.compression,
    minify: config.minify,
    sourceMaps: config.sourceMaps,
    analyze: config.analyze
  };
}

/**
 * Generate build configuration summary
 */
function generateConfigSummary(environment = 'production') {
  const config = getConfig(environment);
  
  return {
    environment,
    timestamp: new Date().toISOString(),
    optimizations: {
      minification: config.minify,
      sourceMaps: config.sourceMaps,
      codeSplitting: !!config.splitChunks,
      compression: config.compression,
      bundleAnalysis: config.analyze
    },
    limits: config.assets,
    webpack: {
      mode: config.webpack?.mode || 'production',
      devtool: config.webpack?.devtool || false
    }
  };
}

module.exports = {
  baseOptimizations,
  environmentConfigs,
  getConfig,
  getWebpackConfig,
  getOptimizationSettings,
  generateConfigSummary
};