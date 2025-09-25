import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { useQuery } from 'react-query';
import GenerationPanel from './components/GenerationPanel';
import Header from './components/Header';
import ImageDisplay from './components/ImageDisplay';
import StatusBar from './components/StatusBar';
import { getStatus } from './services/api';

const App: React.FC = () => {
  const [showInitializationDetails, setShowInitializationDetails] = useState(false);
  
  const { data: status, isLoading, error, refetch } = useQuery(
    'status',
    getStatus,
    {
      refetchInterval: (data) => {
        // More frequent updates during initialization
        if (data?.initialization?.status === 'loading_model' || 
            data?.initialization?.status === 'downloading_model' ||
            data?.initialization?.status === 'loading_to_gpu') {
          return 2000; // 2 seconds during model loading
        }
        // If model is loaded, check less frequently
        return data?.model_loaded ? 30000 : 5000;
      },
      refetchOnWindowFocus: false,
      staleTime: 5000, // Shorter stale time for better initialization feedback
      retry: (failureCount, error) => {
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * attemptIndex, 5000),
      onError: (error: any) => {
        // Only show error toast if it's not a connection error
        if (!error?.message?.includes('Network Error') && !error?.code?.includes('ERR_NETWORK')) {
          toast.error('Failed to connect to API server');
        }
      }
    }
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-6">
            <StatusBar 
              status={status} 
              isLoading={isLoading} 
              error={error}
              onRetry={() => refetch()}
            />
            <GenerationPanel />
          </div>
          
          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            <ImageDisplay />
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="container mx-auto px-4 py-6 text-center text-gray-600">
          <p>Qwen-Image Generator v2.0 | Professional AI Image Generation</p>
          <p className="text-sm mt-1">
            Optimized for RTX 4080 • Memory-Managed • React + FastAPI
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;