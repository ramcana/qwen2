const API_BASE_URL = 'http://localhost:8000/api';

export const generateTextToImage = async (params) => {
  const response = await fetch(`${API_BASE_URL}/generate/text-to-image`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate image');
  }

  const imageBlob = await response.blob();
  return URL.createObjectURL(imageBlob);
};

export const generateImageToImage = async (formData) => {
  const response = await fetch(`${API_BASE_URL}/generate/image-to-image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate image');
  }

  const imageBlob = await response.blob();
  return URL.createObjectURL(imageBlob);
};

export const generateInpainting = async (formData) => {
  const response = await fetch(`${API_BASE_URL}/generate/inpainting`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate image');
  }

  const imageBlob = await response.blob();
  return URL.createObjectURL(imageBlob);
};

export const generateSuperResolution = async (formData) => {
  const response = await fetch(`${API_BASE_URL}/generate/super-resolution`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate image');
  }

  const imageBlob = await response.blob();
  return URL.createObjectURL(imageBlob);
};
