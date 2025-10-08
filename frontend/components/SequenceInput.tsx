import { useState, useEffect } from 'react';
import TextareaAutosize from 'react-textarea-autosize';
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import { SequenceInputProps } from '../types';

const SequenceInput: React.FC<SequenceInputProps> = ({
  value,
  onChange,
  placeholder = "Enter your genome sequence here (e.g., ATCGATCGATCG...)",
  disabled = false,
  maxLength = 10000
}) => {
  const [stats, setStats] = useState({
    length: 0,
    gcContent: 0,
    validNucleotides: true
  });

  // Calculate sequence statistics
  useEffect(() => {
    if (!value) {
      setStats({ length: 0, gcContent: 0, validNucleotides: true });
      return;
    }

    const cleanSequence = value.replace(/\s/g, '').toUpperCase();
    const length = cleanSequence.length;
    
    // Calculate GC content
    const gcCount = (cleanSequence.match(/[GC]/g) || []).length;
    const gcContent = length > 0 ? (gcCount / length) * 100 : 0;
    
    // Check if sequence contains only valid nucleotides
    const validNucleotides = /^[ATCGN]*$/.test(cleanSequence);
    
    setStats({
      length,
      gcContent,
      validNucleotides
    });
  }, [value]);

  const getSequenceValidation = () => {
    if (!value.trim()) return null;
    
    if (!stats.validNucleotides) {
      return {
        type: 'error',
        message: 'Sequence contains invalid characters. Only A, T, C, G, N are allowed.'
      };
    }
    
    if (stats.length < 10) {
      return {
        type: 'warning',
        message: 'Sequence is very short. Consider using sequences of at least 50 nucleotides for better predictions.'
      };
    }
    
    if (stats.length > 5000) {
      return {
        type: 'info',
        message: 'Long sequences may take more time to analyze.'
      };
    }
    
    return null;
  };

  const validation = getSequenceValidation();

  return (
    <div className="space-y-4">
      <div className="relative">
        <label htmlFor="sequence-input" className="block text-sm font-medium text-gray-700 mb-2">
          Genome Sequence
        </label>
        
        <TextareaAutosize
          id="sequence-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          maxLength={maxLength}
          minRows={4}
          maxRows={10}
          className={`textarea font-mono text-sm ${
            validation?.type === 'error' ? 'border-danger-300 focus:ring-danger-500 focus:border-danger-500' : ''
          } ${disabled ? 'bg-gray-50 cursor-not-allowed' : ''}`}
          style={{ resize: 'vertical' }}
        />
        
        {/* Character count */}
        <div className="absolute bottom-2 right-2 text-xs text-gray-400 bg-white px-1 rounded">
          {stats.length} / {maxLength}
        </div>
      </div>

      {/* Sequence Statistics */}
      {value && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">{stats.length.toLocaleString()}</div>
            <div className="text-sm text-gray-500">Nucleotides</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">{stats.gcContent.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">GC Content</div>
          </div>
          <div className="text-center">
            <div className={`text-lg font-semibold ${stats.validNucleotides ? 'text-secondary-600' : 'text-danger-600'}`}>
              {stats.validNucleotides ? '✓ Valid' : '✗ Invalid'}
            </div>
            <div className="text-sm text-gray-500">Format</div>
          </div>
        </div>
      )}

      {/* Validation Message */}
      {validation && (
        <div className={`flex items-start space-x-2 p-3 rounded-lg ${
          validation.type === 'error' ? 'bg-danger-50 border border-danger-200' :
          validation.type === 'warning' ? 'bg-warning-50 border border-warning-200' :
          'bg-primary-50 border border-primary-200'
        }`}>
          <InformationCircleIcon className={`h-5 w-5 mt-0.5 flex-shrink-0 ${
            validation.type === 'error' ? 'text-danger-500' :
            validation.type === 'warning' ? 'text-warning-500' :
            'text-primary-500'
          }`} />
          <p className={`text-sm ${
            validation.type === 'error' ? 'text-danger-700' :
            validation.type === 'warning' ? 'text-warning-700' :
            'text-primary-700'
          }`}>
            {validation.message}
          </p>
        </div>
      )}

      {/* Format Guidelines */}
      <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded-lg">
        <h4 className="font-medium text-gray-700 mb-2">Format Guidelines:</h4>
        <ul className="space-y-1 list-disc list-inside">
          <li>Use standard nucleotide codes: A, T, C, G, N</li>
          <li>Whitespace and line breaks will be automatically removed</li>
          <li>Both uppercase and lowercase letters are accepted</li>
          <li>Optimal sequence length: 50-1000 nucleotides</li>
        </ul>
      </div>
    </div>
  );
};

export default SequenceInput;