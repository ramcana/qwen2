import { Image as ImageIcon, Upload, X } from "lucide-react";
import React, { useRef, useState } from "react";
import { toast } from "react-hot-toast";

interface ImageUploadProps {
  onImageUpload: (file: File, previewUrl: string) => void;
  currentImage?: string;
  onClearImage?: () => void;
  disabled?: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageUpload,
  currentImage,
  onClearImage,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find((file) => file.type.startsWith("image/"));

    if (imageFile) {
      handleImageFile(imageFile);
    } else {
      toast.error("Please upload an image file");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleImageFile(file);
    }
  };

  const handleImageFile = (file: File) => {
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast.error("Image must be smaller than 10MB");
      return;
    }

    // Validate file type
    if (!file.type.startsWith("image/")) {
      toast.error("Please select an image file");
      return;
    }

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    onImageUpload(file, previewUrl);
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="space-y-3">
      <label className="playground-label">Input Image</label>

      {currentImage ? (
        <div className="relative group">
          <img
            src={currentImage}
            alt="Input"
            className="w-full h-48 object-cover rounded-xl border border-slate-200"
          />
          {onClearImage && (
            <button
              onClick={onClearImage}
              className="absolute top-2 right-2 btn btn-secondary p-2 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Remove image"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      ) : (
        <div
          className={`
            relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer
            ${isDragging ? "border-violet-400 bg-violet-50" : "border-slate-300 hover:border-slate-400"}
            ${disabled ? "opacity-50 cursor-not-allowed" : "hover:bg-slate-50"}
          `}
          onDrop={handleDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onClick={handleClick}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={disabled}
          />

          <div className="space-y-3">
            <div className="flex justify-center">
              <div className="p-3 bg-slate-100 rounded-full">
                {isDragging ? (
                  <Upload className="w-8 h-8 text-violet-600" />
                ) : (
                  <ImageIcon className="w-8 h-8 text-slate-400" />
                )}
              </div>
            </div>

            <div>
              <p className="text-sm font-medium text-slate-700">
                {isDragging
                  ? "Drop image here"
                  : "Click to upload or drag & drop"}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                PNG, JPG, WebP up to 10MB
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
