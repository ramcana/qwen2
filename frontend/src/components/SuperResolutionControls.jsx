import React, { useState, useEffect } from 'react';
import Slider from './Slider';

const SuperResolutionControls = ({ setGenerationParams }) => {
  const [inputImage, setInputImage] = useState(null);
  const [scaleFactor, setScaleFactor] = useState(2);

  useEffect(() => {
    setGenerationParams({
      inputImage,
      scaleFactor,
    });
  }, [inputImage, scaleFactor, setGenerationParams]);

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Image to Enhance</label>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setInputImage(e.target.files[0])}
          className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
        />
      </div>

      <Slider label="Scale Factor" value={scaleFactor} onChange={(e) => setScaleFactor(e.target.value)} min={2} max={4} step={1} />
    </div>
  );
};

export default SuperResolutionControls;
