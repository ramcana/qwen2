import React from 'react';

const Slider = ({ label, value, onChange, min, max, step }) => {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <div className="flex items-center gap-4">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
        />
        <span className="text-sm font-semibold text-cyan-400 w-16 text-center">{value}</span>
      </div>
    </div>
  );
};

export default Slider;
