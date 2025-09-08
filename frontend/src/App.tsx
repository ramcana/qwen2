import React from 'react';
import { toast } from 'react-hot-toast';
import { useQuery } from 'react-query';
import GenerationPanel from './components/GenerationPanel';
import Header from './components/Header';
import ImageDisplay from './components/ImageDisplay';
import StatusBar from './components/StatusBar';
import { getStatus } from './services/api';

const App: React.FC = () => {
  const { data: status, isLoading } = useQuery(
    'status',
    getStatus,
    {
      refetchInterval: 10000, // Refresh every 10 seconds
      onError: () => {
        toast.error('Failed to connect to API server');
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
            <StatusBar status={status} isLoading={isLoading} />
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