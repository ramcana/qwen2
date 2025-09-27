import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, Image as ImageIcon } from "lucide-react";

interface ImageUploadAreaProps {
  label: string;
  accept?: string;
  onFileSelect: (file: File | null) => void;
  currentFile?: File;
  placeholder?: string;
  required?: boolean;
  disabled?: boolean;
}

const ImageUploadArea: React.FC<ImageUploadAreaProps> = ({
  label,
  accept = "image/*",
  onFileSelect,
  currentFile,
  placeholder = "Drop an image here or click to browse",
  required = false,
  disabled = false,
}) => {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        onFileSelect(file);

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
          setPreview(e.target?.result as string);
        };
        reader.readAsDataURL(file);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { [accept]: [] },
    multiple: false,
    disabled,
  });

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileSelect(null);
    setPreview(null);
  };

  // Set preview when currentFile changes externally
  React.useEffect(() => {
    if (currentFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(currentFile);
    } else {
      setPreview(null);
    }
  }, [currentFile]);

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>

      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-6 transition-colors cursor-pointer
          ${
            isDragActive
              ? "border-blue-400 bg-blue-50"
              : currentFile
                ? "border-green-400 bg-green-50"
                : "border-gray-300 bg-gray-50 hover:border-gray-400"
          }
          ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        <input {...getInputProps()} />

        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="max-w-full max-h-48 mx-auto rounded-lg shadow-sm"
            />
            <button
              onClick={handleRemove}
              className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
              title="Remove image"
            >
              <X size={16} />
            </button>
            <div className="mt-2 text-sm text-gray-600 text-center">
              {currentFile?.name}
            </div>
          </div>
        ) : (
          <div className="text-center">
            {isDragActive ? (
              <Upload className="mx-auto h-12 w-12 text-blue-400" />
            ) : (
              <ImageIcon className="mx-auto h-12 w-12 text-gray-400" />
            )}
            <div className="mt-2">
              <p className="text-sm text-gray-600">
                {isDragActive ? "Drop the image here" : placeholder}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Supports: PNG, JPG, JPEG, WebP
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploadArea;
