import React, { useState, useEffect } from 'react';
import Slider from './Slider';
import { styles } from '../styles';

const ImageToImageControls = ({ setGenerationParams, selectedStyle }) => {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [strength, setStrength] = useState(0.8);
  const [initImage, setInitImage] = useState(null);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(50);
  const [cfgScale, setCfgScale] = useState(7.0);
  const [seed, setSeed] = useState(-1);
  const [enhancePrompt, setEnhancePrompt] = useState(true);

  useEffect(() => {
    const style_info = styles.find(s => s.name === selectedStyle);
    const final_prompt = style_info.prompt.replace('{prompt}', prompt);

    setGenerationParams({
      prompt: final_prompt,
      negativePrompt,
      strength,
      initImage,
      width,
      height,
      steps,
      cfgScale,
      seed,
      enhancePrompt,
      style: selectedStyle,
    });
  }, [prompt, negativePrompt, strength, initImage, width, height, steps, cfgScale, seed, enhancePrompt, selectedStyle, setGenerationParams]);

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Initial Image</label>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setInitImage(e.target.files[0])}
          className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white h-24 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          placeholder="A beautiful landscape painting..."
        />
      </div>
      <Slider label="Strength" value={strength} onChange={(e) => setStrength(e.target.value)} min={0.1} max={1.0} step={0.05} />
      <Slider label="Width" value={width} onChange={(e) => setWidth(e.target.value)} min={512} max={2048} step={64} />
      <Slider label="Height" value={height} onChange={(e) => setHeight(e.target.value)} min={512} max={2048} step={64} />
      <Slider label="Inference Steps" value={steps} onChange={(e) => setSteps(e.target.value)} min={10} max={100} step={5} />
      <Slider label="CFG Scale" value={cfgScale} onChange={(e) => setCfgScale(e.target.value)} min={1.0} max={20.0} step={0.5} />

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Seed</label>
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(parseInt(e.target.value))}
          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
        />
      </div>

      <div className="flex items-center">
        <input
          id="enhance-prompt-i2i"
          type="checkbox"
          checked={enhancePrompt}
          onChange={(e) => setEnhancePrompt(e.target.checked)}
          className="h-4 w-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
        />
        <label htmlFor="enhance-prompt-i2i" className="ml-2 block text-sm text-gray-300">
          Enhance Prompt
        </label>
      </div>
    </div>
  );
};

export default ImageToImageControls;
