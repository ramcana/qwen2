import React, { useState, useRef, useEffect } from 'react';
import { MaskEditor } from 'react-mask-editor';
import Slider from './Slider';
import { styles } from '../styles';

const InpaintingControls = ({ setGenerationParams, selectedStyle }) => {
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [initImage, setInitImage] = useState(null); // for display
  const [initImageData, setInitImageData] = useState(null); // for upload
  const [mask, setMask] = useState(null);
  const maskEditorRef = useRef(null);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(50);
  const [cfgScale, setCfgScale] = useState(7.0);
  const [seed, setSeed] = useState(-1);
  const [enhancePrompt, setEnhancePrompt] = useState(true);

  useEffect(() => {
    const style_info = styles.find(s => s.name === selectedStyle);
    const final_prompt = style_info ? style_info.prompt.replace('{prompt}', prompt) : prompt;

    setGenerationParams({
      prompt: final_prompt,
      negativePrompt,
      initImage: initImageData,
      maskImage: mask,
      width,
      height,
      steps,
      cfgScale,
      seed,
      enhancePrompt,
      style: selectedStyle,
    });
  }, [prompt, negativePrompt, initImageData, mask, width, height, steps, cfgScale, seed, enhancePrompt, selectedStyle, setGenerationParams]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setInitImageData(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setInitImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSaveMask = () => {
    if (maskEditorRef.current) {
      const maskCanvas = maskEditorRef.current.getMaskCanvas();
      maskCanvas.toBlob(blob => setMask(blob));
      alert("Mask has been prepared for generation.");
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Initial Image</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
        />
      </div>

      {initImage && (
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Draw Mask</label>
          <div className="border border-gray-600 rounded-md overflow-hidden">
            <MaskEditor
              ref={maskEditorRef}
              image={initImage}
              width={400}
              height={400}
            />
          </div>
          <button
            onClick={handleSaveMask}
            className="mt-2 bg-gray-600 hover:bg-gray-500 text-white font-bold py-1 px-3 rounded-md text-sm"
          >
            Use This Mask
          </button>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white h-24 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          placeholder="A futuristic city in the masked area..."
        />
      </div>

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
          id="enhance-prompt-inpaint"
          type="checkbox"
          checked={enhancePrompt}
          onChange={(e) => setEnhancePrompt(e.target.checked)}
          className="h-4 w-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
        />
        <label htmlFor="enhance-prompt-inpaint" className="ml-2 block text-sm text-gray-300">
          Enhance Prompt
        </label>
      </div>
    </div>
  );
};

export default InpaintingControls;
