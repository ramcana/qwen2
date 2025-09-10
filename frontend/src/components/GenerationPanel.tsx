import { Settings, Shuffle, Wand2 } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { toast } from 'react-hot-toast';
import { useMutation, useQuery } from 'react-query';
import { generateTextToImage, getAspectRatios } from '../services/api';
import { GenerationRequest } from '../types/api';

interface GenerationFormData extends Omit<GenerationRequest, 'seed'> {
  seed: string; // Form uses string, convert to number
}

const GenerationPanel: React.FC = () => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [lastGeneration, setLastGeneration] = useState<any>(null);

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<GenerationFormData>({
    defaultValues: {
      prompt: '',
      negative_prompt: '',
      width: 1664,
      height: 928,
      num_inference_steps: 50,
      cfg_scale: 4.0,
      seed: '-1',
      language: 'en',
      enhance_prompt: true,
      aspect_ratio: '16:9'
    }
  });

  const { data: aspectRatios } = useQuery('aspect-ratios', getAspectRatios);

  const selectedAspectRatio = watch('aspect_ratio');

  // Update dimensions when aspect ratio changes
  useEffect(() => {
    if (aspectRatios && selectedAspectRatio && aspectRatios.ratios[selectedAspectRatio]) {
      const [width, height] = aspectRatios.ratios[selectedAspectRatio];
      setValue('width', width);
      setValue('height', height);
    }
  }, [selectedAspectRatio, aspectRatios, setValue]);

  const generateMutation = useMutation(generateTextToImage, {
    onSuccess: (data) => {
      if (data.success) {
        setLastGeneration(data);
        toast.success(`Image generated in ${data.generation_time?.toFixed(1)}s`);
      } else {
        toast.error(data.message);
      }
    },
    onError: (error: any) => {
      toast.error(`Generation failed: ${error.response?.data?.detail || error.message}`);
    }
  });

  const onSubmit = (data: GenerationFormData) => {
    const request: GenerationRequest = {
      ...data,
      seed: data.seed === '-1' ? -1 : parseInt(data.seed) || -1
    };
    generateMutation.mutate(request);
  };

  const randomizeSeed = () => {
    setValue('seed', Math.floor(Math.random() * 1000000).toString());
  };

  const quickPresets = [
    { name: 'Fast Preview', steps: 20, cfg: 3.0 },
    { name: 'Balanced', steps: 50, cfg: 4.0 },
    { name: 'High Quality', steps: 80, cfg: 7.0 }
  ];

  const examplePrompts = [
    "A futuristic coffee shop with neon signs reading 'AI Café' and 'Welcome' in both English and Chinese, cyberpunk style",
    "A beautiful landscape painting with text overlay reading 'Qwen Mountain Resort - Est. 2025', traditional Chinese painting style",
    "A modern poster design with the text 'Innovation Summit 2025' in bold letters, minimalist design, blue and white color scheme"
  ];

  return (
    <div className="space-y-6">
      {/* Generation Form */}
      <div className="card p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Generate Image</h2>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Prompt */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prompt
            </label>
            <textarea
              {...register('prompt', { required: 'Prompt is required' })}
              className="input min-h-[100px] resize-none"
              placeholder="A coffee shop entrance with a chalkboard sign reading 'Qwen Coffee ☕ $2 per cup'..."
            />
            {errors.prompt && (
              <p className="text-red-500 text-sm mt-1">{errors.prompt.message}</p>
            )}
          </div>

          {/* Language and Enhancement */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Language
              </label>
              <select {...register('language')} className="input">
                <option value="en">English</option>
                <option value="zh">中文</option>
              </select>
            </div>
            <div className="flex items-center pt-8">
              <input
                {...register('enhance_prompt')}
                type="checkbox"
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label className="ml-2 text-sm text-gray-700">
                Enhance prompt
              </label>
            </div>
          </div>

          {/* Aspect Ratio and Dimensions */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Aspect Ratio
            </label>
            <select {...register('aspect_ratio')} className="input">
              {aspectRatios && Object.keys(aspectRatios.ratios).map(ratio => (
                <option key={ratio} value={ratio}>
                  {ratio.replace(':', ':')} ({aspectRatios.ratios[ratio][0]}×{aspectRatios.ratios[ratio][1]})
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Width
              </label>
              <input
                {...register('width', { min: 512, max: 2048 })}
                type="number"
                step="64"
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Height
              </label>
              <input
                {...register('height', { min: 512, max: 2048 })}
                type="number"
                step="64"
                className="input"
              />
            </div>
          </div>

          {/* Advanced Settings */}
          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center text-sm text-gray-600 hover:text-gray-900"
            >
              <Settings className="w-4 h-4 mr-1" />
              Advanced Settings
            </button>
          </div>

          {showAdvanced && (
            <div className="space-y-4 pt-4 border-t">
              {/* Negative Prompt */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Negative Prompt
                </label>
                <textarea
                  {...register('negative_prompt')}
                  className="input min-h-[60px] resize-none"
                  placeholder="blurry, low quality, distorted..."
                />
              </div>

              {/* Steps and CFG */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Steps: {watch('num_inference_steps')}
                  </label>
                  <input
                    {...register('num_inference_steps', { min: 10, max: 100 })}
                    type="range"
                    min="10"
                    max="100"
                    step="5"
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    CFG Scale: {watch('cfg_scale')}
                  </label>
                  <input
                    {...register('cfg_scale', { min: 1, max: 20 })}
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    className="w-full"
                  />
                </div>
              </div>

              {/* Seed */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Seed
                </label>
                <div className="flex space-x-2">
                  <input
                    {...register('seed')}
                    className="input flex-1"
                    placeholder="-1 for random"
                  />
                  <button
                    type="button"
                    onClick={randomizeSeed}
                    className="btn btn-secondary px-3"
                  >
                    <Shuffle className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Quick Presets */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Presets
            </label>
            <div className="flex space-x-2">
              {quickPresets.map(preset => (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => {
                    setValue('num_inference_steps', preset.steps);
                    setValue('cfg_scale', preset.cfg);
                  }}
                  className="btn btn-secondary text-xs px-3 py-1"
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>

          {/* Generate Button */}
          <button
            type="submit"
            disabled={generateMutation.isLoading}
            className="w-full btn btn-primary py-3 text-lg"
          >
            {generateMutation.isLoading ? (
              <div className="flex items-center justify-center">
                <Wand2 className="w-5 h-5 mr-2 animate-spin" />
                Generating...
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <Wand2 className="w-5 h-5 mr-2" />
                Generate Image
              </div>
            )}
          </button>
        </form>
      </div>

      {/* Example Prompts */}
      <div className="card p-4">
        <h3 className="font-medium text-gray-900 mb-3">Example Prompts</h3>
        <div className="space-y-2">
          {examplePrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => setValue('prompt', prompt)}
              className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border text-gray-700"
            >
              {prompt.length > 80 ? `${prompt.substring(0, 80)}...` : prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default GenerationPanel;
