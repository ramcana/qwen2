import { useState } from 'react';
import TextToImageControls from './components/TextToImageControls';
import ImageToImageControls from './components/ImageToImageControls';
import InpaintingControls from './components/InpaintingControls';
import SuperResolutionControls from './components/SuperResolutionControls';
import StylePicker from './components/StylePicker';
import {
  generateTextToImage,
  generateImageToImage,
  generateInpainting,
  generateSuperResolution,
} from './api';

function App() {
  const [mode, setMode] = useState('text-to-image');
  const [generationParams, setGenerationParams] = useState({});
  const [selectedStyle, setSelectedStyle] = useState('Default');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    setGeneratedImage(null);

    try {
      let imageUrl;
      if (mode === 'text-to-image') {
        imageUrl = await generateTextToImage(generationParams);
      } else {
        const formData = new FormData();
        Object.keys(generationParams).forEach(key => {
          // Do not append image files here, they are handled below
          if (key !== 'initImage' && key !== 'maskImage' && key !== 'inputImage') {
            formData.append(key, generationParams[key]);
          }
        });

        if (mode === 'image-to-image') {
          formData.append('init_image', generationParams.initImage);
          imageUrl = await generateImageToImage(formData);
        } else if (mode === 'inpainting') {
          formData.append('init_image', generationParams.initImageData); // use the raw file
          formData.append('mask_image', generationParams.maskImage);
          imageUrl = await generateInpainting(formData);
        } else if (mode === 'super-resolution') {
          formData.append('input_image', generationParams.inputImage);
          imageUrl = await generateSuperResolution(formData);
        }
      }
      setGeneratedImage(imageUrl);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const renderControls = () => {
    const props = {
      setGenerationParams,
      selectedStyle,
    };
    switch (mode) {
      case 'text-to-image':
        return <TextToImageControls {...props} />;
      case 'image-to-image':
        return <ImageToImageControls {...props} />;
      case 'inpainting':
        return <InpaintingControls {...props} />;
      case 'super-resolution':
        return <SuperResolutionControls setGenerationParams={setGenerationParams} />;
      default:
        return null;
    }
  };

  return (
    <div className="bg-gray-900 text-white min-h-screen font-sans">
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4 py-3">
          <h1 className="text-2xl font-bold text-cyan-400">Qwen Image Generator</h1>
          <p className="text-sm text-gray-400">Powered by FastAPI & React</p>
        </div>
      </header>

      <div className="flex container mx-auto px-4 py-6 gap-6">
        <aside className="w-1/3 bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold mb-4 border-b border-gray-600 pb-2">Controls</h2>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Generation Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <option value="text-to-image">Text-to-Image</option>
              <option value="image-to-image">Image-to-Image</option>
              <option value="inpainting">Inpainting</option>
              <option value="super-resolution">Super Resolution</option>
            </select>
          </div>

          { (mode !== 'super-resolution') &&
            <div className="mb-6">
              <StylePicker selectedStyle={selectedStyle} onStyleChange={setSelectedStyle} />
            </div>
          }

          {renderControls()}
        </aside>

        <main className="w-2/3 bg-gray-800 p-6 rounded-lg shadow-lg flex flex-col items-center justify-center">
          <div className="w-full h-full border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center bg-gray-900/50">
            {isLoading && <div className="text-white">Generating...</div>}
            {error && <div className="text-red-500 p-4">{error}</div>}
            {generatedImage && !isLoading && !error && (
              <img src={generatedImage} alt="Generated" className="max-w-full max-h-full object-contain" />
            )}
            {!generatedImage && !isLoading && !error && (
              <p className="text-gray-500">Generated image will appear here</p>
            )}
          </div>
          <button
            onClick={handleGenerate}
            disabled={isLoading}
            className="mt-6 bg-cyan-500 hover:bg-cyan-600 text-white font-bold py-2 px-6 rounded-lg shadow-md transition-colors duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Generating...' : 'Generate'}
          </button>
        </main>
      </div>
    </div>
  );
}

export default App;
