import React, { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, Image as ImageIcon, Eye } from "lucide-react";

interface ControlImageUploadProps {
  onFileSelect: (file: File | null) => void;
  currentFile?: File;
  disabled?: boolean;
  showPreview?: boolean;
  onPreviewToggle?: () => void;
}

const ControlImageUpload: React.FC<ControlImageUploadProps> = ({
  onFileSelect,
  currentFile,
  disabled = false,
  showPreview = false,
  onPreviewToggle,
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
    accept: { "image/*": [] },
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
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          Control Image
          <span className="text-red-500 ml-1">*</span>
        </label>
        {currentFile && onPreviewToggle && (
          <button
            onClick={onPreviewToggle}
            className="flex items-center space-x-1 text-sm text-purple-600 hover:text-purple-700"
          >
            <Eye size={14} />
            <span>{showPreview ? "Hide" : "Show"} Control Preview</span>
          </button>
        )}
      </div>

      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-6 transition-colors cursor-pointer
          ${
            isDragActive
              ? "border-purple-400 bg-purple-50"
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
              alt="Control Image Preview"
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
              <Upload className="mx-auto h-12 w-12 text-purple-400" />
            ) : (
              <ImageIcon className="mx-auto h-12 w-12 text-gray-400" />
            )}
            <div className="mt-2">
              <p className="text-sm text-gray-600">
                {isDragActive
                  ? "Drop the control image here"
                  : "Drop a control image here or click to browse"}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                This image will guide the generation process
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlImageUpload;
