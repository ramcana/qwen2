import React from 'react';
import { styles } from '../styles';

const StylePicker = ({ selectedStyle, onStyleChange }) => {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">Style</label>
      <select
        value={selectedStyle}
        onChange={(e) => onStyleChange(e.target.value)}
        className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
      >
        {styles.map((style) => (
          <option key={style.name} value={style.name}>
            {style.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default StylePicker;
