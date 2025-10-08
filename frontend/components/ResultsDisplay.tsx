import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChevronDownIcon, 
  ChevronUpIcon,
  ClockIcon,
  DocumentTextIcon,
  ChartBarIcon,
  InformationCircleIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import { ResultsDisplayProps, PathogenicityClass } from '../types';

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
  const [showDetails, setShowDetails] = useState(false);

  const getPathogenicityBadgeClass = (pathogenicityType: PathogenicityClass): string => {
    const baseClasses = 'badge text-xs font-semibold';
    
    switch (pathogenicityType) {
      case 'benign':
        return `${baseClasses} badge-benign`;
      case 'likely_benign':
        return `${baseClasses} badge-likely-benign`;
      case 'uncertain_significance':
        return `${baseClasses} badge-uncertain`;
      case 'likely_pathogenic':
        return `${baseClasses} badge-likely-pathogenic`;
      case 'pathogenic':
        return `${baseClasses} badge-pathogenic`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const getScoreBarColor = (score: number): string => {
    if (score < 0.2) return 'bg-secondary-500';
    if (score < 0.4) return 'bg-secondary-400';
    if (score < 0.6) return 'bg-warning-500';
    if (score < 0.8) return 'bg-danger-400';
    return 'bg-danger-500';
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence > 0.8) return 'text-secondary-600';
    if (confidence > 0.6) return 'text-warning-600';
    return 'text-danger-600';
  };

  const formatTimestamp = (timestamp: Date): string => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(timestamp);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="card"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div className="flex items-start space-x-4">
          <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center flex-shrink-0">
            <BeakerIcon className="h-5 w-5 text-primary-600" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center space-x-3 mb-1">
              <h3 className="text-lg font-semibold text-gray-900">Analysis Result</h3>
              <span className={getPathogenicityBadgeClass(result.prediction.pathogenicity_type)}>
                {result.prediction.pathogenicity_type.replace('_', ' ')}
              </span>
            </div>
            <div className="flex items-center text-sm text-gray-500 space-x-4">
              <div className="flex items-center">
                <ClockIcon className="h-4 w-4 mr-1" />
                {formatTimestamp(result.timestamp)}
              </div>
              {result.fileName && (
                <div className="flex items-center">
                  <DocumentTextIcon className="h-4 w-4 mr-1" />
                  {result.fileName}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Results */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Pathogenicity Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Pathogenicity Score</span>
            <span className="text-lg font-bold text-gray-900">
              {(result.prediction.pathogenicity_score * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${result.prediction.pathogenicity_score * 100}%` }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className={`h-2 rounded-full ${getScoreBarColor(result.prediction.pathogenicity_score)}`}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>Benign</span>
            <span>Pathogenic</span>
          </div>
        </div>

        {/* Confidence */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Confidence</span>
            <span className={`text-lg font-bold ${getConfidenceColor(result.prediction.confidence)}`}>
              {(result.prediction.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${result.prediction.confidence * 100}%` }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="bg-primary-500 h-2 rounded-full"
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>

        {/* Sequence Info */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700">Sequence Information</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Length:</span>
              <span className="font-medium text-gray-900">
                {result.prediction.sequence_info.length.toLocaleString()} bp
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">GC Content:</span>
              <span className="font-medium text-gray-900">
                {(result.prediction.sequence_info.gc_content * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Sequence Preview */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Sequence Preview</h4>
        <div className="bg-gray-50 rounded-lg p-3 font-mono text-sm text-gray-700 break-all border">
          {result.sequence}
        </div>
      </div>

      {/* Model Info */}
      <div className="flex items-center justify-between py-3 border-t border-gray-200">
        <div className="flex items-center text-sm text-gray-600">
          <InformationCircleIcon className="h-4 w-4 mr-2" />
          <span>Model: {result.prediction.model_info.name} v{result.prediction.model_info.version}</span>
        </div>
        
        {/* Toggle Details Button */}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center text-sm text-primary-600 hover:text-primary-700 transition-colors duration-200"
        >
          <ChartBarIcon className="h-4 w-4 mr-1" />
          {showDetails ? 'Hide Details' : 'Show Details'}
          {showDetails ? (
            <ChevronUpIcon className="h-4 w-4 ml-1" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 ml-1" />
          )}
        </button>
      </div>

      {/* Detailed Results */}
      {showDetails && result.prediction.detailed_results && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3 }}
          className="border-t border-gray-200 pt-6 space-y-6"
        >
          {/* Class Probabilities */}
          {result.prediction.detailed_results.prob_benign !== undefined && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Class Probabilities</h4>
              <div className="space-y-3">
                {[
                  { key: 'prob_benign', label: 'Benign', color: 'bg-secondary-500' },
                  { key: 'prob_likely_benign', label: 'Likely Benign', color: 'bg-secondary-400' },
                  { key: 'prob_uncertain_significance', label: 'Uncertain', color: 'bg-warning-500' },
                  { key: 'prob_likely_pathogenic', label: 'Likely Pathogenic', color: 'bg-danger-400' },
                  { key: 'prob_pathogenic', label: 'Pathogenic', color: 'bg-danger-500' },
                ].map((prob) => {
                  const value = result.prediction.detailed_results![prob.key] as number || 0;
                  return (
                    <div key={prob.key} className="flex items-center space-x-3">
                      <div className="w-20 text-xs text-gray-600">{prob.label}</div>
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${value * 100}%` }}
                          transition={{ duration: 0.6, delay: 0.1 }}
                          className={`h-2 rounded-full ${prob.color}`}
                        />
                      </div>
                      <div className="w-12 text-xs font-medium text-gray-900 text-right">
                        {(value * 100).toFixed(1)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Feature Importance */}
          {result.prediction.detailed_results.top_features && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Top Contributing Features</h4>
              <div className="space-y-2">
                {result.prediction.detailed_results.top_features.slice(0, 8).map((feature, index) => (
                  <div key={index} className="flex items-center justify-between py-2 px-3 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700 font-mono">{feature.feature}</span>
                    <span className="text-sm font-medium text-gray-900">
                      {(feature.importance * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
};

export default ResultsDisplay;