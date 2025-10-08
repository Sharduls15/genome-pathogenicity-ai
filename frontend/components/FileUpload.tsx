import { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  DocumentArrowUpIcon, 
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { FileUploadProps } from '../types';

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  acceptedFormats,
  maxSize,
  disabled = false
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxSize) {
      return `File size (${formatFileSize(file.size)}) exceeds maximum allowed size (${formatFileSize(maxSize)})`;
    }

    // Check file extension
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    const isValidFormat = acceptedFormats.some(format => 
      format.toLowerCase() === fileExtension
    );

    if (!isValidFormat) {
      return `Invalid file format. Accepted formats: ${acceptedFormats.join(', ')}`;
    }

    return null;
  };

  const handleFiles = useCallback((files: FileList) => {
    if (files.length === 0) return;

    const file = files[0];
    const validationError = validateFile(file);

    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }

    setError(null);
    setSelectedFile(file);
    onFileSelect(file);
  }, [onFileSelect, maxSize, acceptedFormats]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles, disabled]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  };

  const handleButtonClick = () => {
    if (inputRef.current) {
      inputRef.current.click();
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
          dragActive
            ? 'border-primary-400 bg-primary-50'
            : error
            ? 'border-danger-300 bg-danger-50'
            : selectedFile
            ? 'border-secondary-300 bg-secondary-50'
            : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!disabled ? handleButtonClick : undefined}
      >
        <input
          ref={inputRef}
          type="file"
          accept={acceptedFormats.join(',')}
          onChange={handleChange}
          disabled={disabled}
          className="hidden"
        />

        <div className="space-y-4">
          {selectedFile ? (
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex flex-col items-center"
            >
              <CheckCircleIcon className="h-12 w-12 text-secondary-500" />
              <div className="text-sm font-medium text-gray-900">
                {selectedFile.name}
              </div>
              <div className="text-xs text-gray-500">
                {formatFileSize(selectedFile.size)}
              </div>
            </motion.div>
          ) : error ? (
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex flex-col items-center"
            >
              <ExclamationTriangleIcon className="h-12 w-12 text-danger-500" />
              <div className="text-sm font-medium text-danger-700">
                Upload Error
              </div>
            </motion.div>
          ) : (
            <div className="flex flex-col items-center">
              <DocumentArrowUpIcon className="h-12 w-12 text-gray-400" />
              <div className="text-sm font-medium text-gray-900">
                Drop files here or click to browse
              </div>
              <div className="text-xs text-gray-500">
                {acceptedFormats.join(', ')} up to {formatFileSize(maxSize)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center space-x-2 p-3 bg-danger-50 border border-danger-200 rounded-lg"
        >
          <ExclamationTriangleIcon className="h-5 w-5 text-danger-500 flex-shrink-0" />
          <p className="text-sm text-danger-700">{error}</p>
        </motion.div>
      )}

      {/* Selected File Info */}
      {selectedFile && !error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between p-3 bg-secondary-50 border border-secondary-200 rounded-lg"
        >
          <div className="flex items-center space-x-3">
            <CheckCircleIcon className="h-5 w-5 text-secondary-600" />
            <div>
              <div className="text-sm font-medium text-gray-900">
                {selectedFile.name}
              </div>
              <div className="text-xs text-gray-500">
                {formatFileSize(selectedFile.size)} • Ready for analysis
              </div>
            </div>
          </div>
          <button
            onClick={clearFile}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors duration-200"
            title="Remove file"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        </motion.div>
      )}

      {/* Supported Formats Info */}
      <div className="text-sm text-gray-500 bg-gray-50 p-3 rounded-lg">
        <h4 className="font-medium text-gray-700 mb-2">Supported File Formats:</h4>
        <ul className="space-y-1 list-disc list-inside">
          <li><strong>FASTA (.fasta, .fa):</strong> Standard sequence format with headers</li>
          <li><strong>Text (.txt):</strong> Plain text containing sequence data</li>
          <li><strong>Maximum file size:</strong> {formatFileSize(maxSize)}</li>
          <li><strong>Multiple sequences:</strong> Only the first sequence will be analyzed</li>
        </ul>
      </div>
    </div>
  );
};

export default FileUpload;