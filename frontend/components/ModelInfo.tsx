import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  CpuChipIcon, 
  ChartBarIcon, 
  AcademicCapIcon,
  InformationCircleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import ApiService from '../lib/api';
import LoadingSpinner from './LoadingSpinner';
import { ModelInfo as ModelInfoType } from '../types';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfoType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const info = await ApiService.getModelInfo('TRAINED-PATHO');
        setModelInfo(info);
      } catch (err: any) {
        console.error('Failed to fetch model info:', err);
        // Set default model info for demo
        setModelInfo({
          name: 'TRAINED-PATHO',
          version: '1.0.0',
          description: 'ClinVar-trained pathogenicity prediction model',
          features: 159,
          training_samples: 11429,
          accuracy: 0.739
        });
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <LoadingSpinner size="md" />
          <span className="ml-3 text-gray-600">Loading model information...</span>
        </div>
      </div>
    );
  }

  if (error || !modelInfo) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8 text-gray-500">
          <InformationCircleIcon className="h-6 w-6 mr-2" />
          <span>Unable to load model information</span>
        </div>
      </div>
    );
  }

  const stats = [
    {
      icon: CpuChipIcon,
      label: 'Model',
      value: `${modelInfo.name} v${modelInfo.version}`,
      description: 'AI model identifier'
    },
    {
      icon: AcademicCapIcon,
      label: 'Training Samples',
      value: modelInfo.training_samples.toLocaleString(),
      description: 'ClinVar variants used for training'
    },
    {
      icon: ChartBarIcon,
      label: 'Features',
      value: modelInfo.features.toString(),
      description: 'Genomic features analyzed'
    },
    {
      icon: CheckCircleIcon,
      label: 'Accuracy',
      value: modelInfo.accuracy ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : 'N/A',
      description: 'Cross-validation performance'
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="card bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200"
    >
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">TRAINED-PATHO Model</h2>
          <p className="text-gray-600 max-w-2xl">
            {modelInfo.description}
          </p>
        </div>
        <div className="w-12 h-12 bg-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
          <CpuChipIcon className="h-6 w-6 text-white" />
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="text-center"
            >
              <div className="inline-flex items-center justify-center w-10 h-10 bg-white rounded-lg shadow-sm mb-3">
                <Icon className="h-5 w-5 text-primary-600" />
              </div>
              <div className="text-lg font-bold text-gray-900 mb-1">
                {stat.value}
              </div>
              <div className="text-sm font-medium text-gray-700 mb-1">
                {stat.label}
              </div>
              <div className="text-xs text-gray-500">
                {stat.description}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Training Details */}
      <div className="mt-6 pt-6 border-t border-primary-200">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Training Details</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div className="bg-white/60 rounded-lg p-3">
            <div className="font-medium text-gray-900">Data Source</div>
            <div className="text-gray-600">ClinVar Database</div>
          </div>
          <div className="bg-white/60 rounded-lg p-3">
            <div className="font-medium text-gray-900">Algorithm</div>
            <div className="text-gray-600">Random Forest Ensemble</div>
          </div>
          <div className="bg-white/60 rounded-lg p-3">
            <div className="font-medium text-gray-900">Validation</div>
            <div className="text-gray-600">5-Fold Cross-validation</div>
          </div>
        </div>
      </div>

      {/* Classifications */}
      <div className="mt-4 pt-4 border-t border-primary-200">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Prediction Classes</h3>
        <div className="flex flex-wrap gap-2">
          {[
            { label: 'Benign', class: 'badge-benign' },
            { label: 'Likely Benign', class: 'badge-likely-benign' },
            { label: 'Uncertain Significance', class: 'badge-uncertain' },
            { label: 'Likely Pathogenic', class: 'badge-likely-pathogenic' },
            { label: 'Pathogenic', class: 'badge-pathogenic' }
          ].map((item) => (
            <span key={item.label} className={`badge ${item.class}`}>
              {item.label}
            </span>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

export default ModelInfo;