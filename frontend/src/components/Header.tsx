import { Cpu, Image, Zap } from 'lucide-react';
import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Image className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Qwen-Image Generator</h1>
              <p className="text-sm text-gray-500">Professional AI Image Generation</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center space-x-2 px-3 py-1 bg-gray-100 rounded-full">
              <Cpu className="w-4 h-4 text-gray-600" />
              <span className="text-sm text-gray-600">RTX 4080</span>
            </div>
            <div className="hidden md:flex items-center space-x-2 px-3 py-1 bg-green-100 rounded-full">
              <Zap className="w-4 h-4 text-green-600" />
              <span className="text-sm text-green-600">Memory Optimized</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
