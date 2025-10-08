import { useState, useCallback } from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { 
  BeakerIcon, 
  DocumentArrowUpIcon, 
  ClipboardDocumentIcon,
  InformationCircleIcon,
  ChartBarIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';

import SequenceInput from '../components/SequenceInput';
import FileUpload from '../components/FileUpload';
import ResultsDisplay from '../components/ResultsDisplay';
import ModelInfo from '../components/ModelInfo';
import Header from '../components/Header';
import Footer from '../components/Footer';
import LoadingSpinner from '../components/LoadingSpinner';

import ApiService from '../lib/api';
import { AnalysisResult, PredictionRequest, ApiError } from '../types';

const HomePage: NextPage = () => {
  const [sequence, setSequence] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [activeTab, setActiveTab] = useState<'text' | 'file'>('text');
  const [selectedModel] = useState('TRAINED-PATHO');

  const handleSequenceAnalysis = useCallback(async (inputSequence: string, inputMethod: 'text' | 'file', fileName?: string) => {
    if (!inputSequence.trim()) {
      toast.error('Please enter a valid genome sequence');
      return;
    }

    // Validate sequence contains only valid nucleotides
    const validNucleotides = /^[ATCGN\s\n\r-]+$/i;
    if (!validNucleotides.test(inputSequence)) {
      toast.error('Sequence contains invalid characters. Only A, T, C, G, N, and whitespace are allowed.');
      return;
    }

    setIsAnalyzing(true);
    const analysisId = Date.now().toString();

    try {
      const request: PredictionRequest = {
        sequence: inputSequence.replace(/\s/g, '').toUpperCase(), // Clean sequence
        model: selectedModel,
        format: 'text',
      };

      const prediction = await ApiService.predictPathogenicity(request);
      
      const newResult: AnalysisResult = {
        id: analysisId,
        sequence: inputSequence.slice(0, 100) + (inputSequence.length > 100 ? '...' : ''), // Truncate for display
        prediction,
        timestamp: new Date(),
        inputMethod,
        fileName,
      };

      setResults(prev => [newResult, ...prev]);
      toast.success('Analysis completed successfully!');

      // Clear input after successful analysis
      if (inputMethod === 'text') {
        setSequence('');
      }

    } catch (error: any) {
      console.error('Analysis failed:', error);
      const apiError = error as ApiError;
      toast.error(apiError.message || 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedModel]);

  const handleTextAnalysis = () => {
    handleSequenceAnalysis(sequence, 'text');
  };

  const handleFileAnalysis = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      if (content) {
        // Simple FASTA parsing - extract sequence data
        let cleanSequence = content;
        if (content.startsWith('>')) {
          // FASTA format - remove header lines
          cleanSequence = content
            .split('\n')
            .filter(line => !line.startsWith('>'))
            .join('');
        }
        handleSequenceAnalysis(cleanSequence, 'file', file.name);
      }
    };
    reader.readAsText(file);
  };

  const exampleSequences = [
    {
      name: 'Normal Sequence',
      sequence: 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC',
      description: 'Balanced nucleotide composition'
    },
    {
      name: 'Low Complexity',
      sequence: 'ATGAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC',
      description: 'Repetitive sequence pattern'
    },
    {
      name: 'High GC Content',
      sequence: 'ATGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
      description: 'GC-rich sequence'
    }
  ];

  return (
    <>
      <Head>
        <title>Genome Pathogenicity AI - TRAINED-PATHO Model</title>
        <meta name="description" content="Analyze genome sequences for pathogenicity using our AI model trained on 75,000 ClinVar variants" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
        <Header />
        
        <main className="container mx-auto px-4 py-8">
          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 rounded-full mb-6">
              <CpuChipIcon className="h-8 w-8 text-primary-600" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Genome Pathogenicity <span className="gradient-text">AI Predictor</span>
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
              Analyze genomic sequences for pathogenicity using our TRAINED-PATHO model, 
              trained on 75,000 expert-curated ClinVar variants with 159 genomic features.
            </p>
            
            <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-500">
              <div className="flex items-center">
                <ChartBarIcon className="h-4 w-4 mr-2" />
                <span>11,429 Training Samples</span>
              </div>
              <div className="flex items-center">
                <BeakerIcon className="h-4 w-4 mr-2" />
                <span>159 Genomic Features</span>
              </div>
              <div className="flex items-center">
                <InformationCircleIcon className="h-4 w-4 mr-2" />
                <span>5-Class Prediction</span>
              </div>
            </div>
          </motion.div>

          {/* Model Info Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mb-8"
          >
            <ModelInfo />
          </motion.div>

          {/* Input Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="card mb-8"
          >
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">Analyze Genome Sequence</h2>
            
            {/* Tab Navigation */}
            <div className="flex space-x-1 mb-6 bg-gray-100 p-1 rounded-lg">
              <button
                onClick={() => setActiveTab('text')}
                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                  activeTab === 'text'
                    ? 'bg-white text-primary-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <ClipboardDocumentIcon className="h-4 w-4 mr-2" />
                Text Input
              </button>
              <button
                onClick={() => setActiveTab('file')}
                className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                  activeTab === 'file'
                    ? 'bg-white text-primary-700 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <DocumentArrowUpIcon className="h-4 w-4 mr-2" />
                File Upload
              </button>
            </div>

            {/* Tab Content */}
            {activeTab === 'text' && (
              <div className="space-y-6">
                <SequenceInput
                  value={sequence}
                  onChange={setSequence}
                  disabled={isAnalyzing}
                />
                
                <div className="flex flex-col sm:flex-row gap-4">
                  <button
                    onClick={handleTextAnalysis}
                    disabled={isAnalyzing || !sequence.trim()}
                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex-1"
                  >
                    {isAnalyzing ? (
                      <>
                        <LoadingSpinner size="sm" className="mr-2" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <BeakerIcon className="h-5 w-5 mr-2" />
                        Analyze Sequence
                      </>
                    )}
                  </button>
                </div>

                {/* Example Sequences */}
                <div className="border-t pt-6">
                  <h3 className="text-sm font-medium text-gray-900 mb-3">Try Example Sequences:</h3>
                  <div className="grid gap-3">
                    {exampleSequences.map((example, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div>
                          <div className="font-medium text-sm text-gray-900">{example.name}</div>
                          <div className="text-xs text-gray-500">{example.description}</div>
                        </div>
                        <button
                          onClick={() => setSequence(example.sequence)}
                          disabled={isAnalyzing}
                          className="btn-secondary text-sm py-1 px-3"
                        >
                          Use
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'file' && (
              <FileUpload
                onFileSelect={handleFileAnalysis}
                acceptedFormats={['.fasta', '.fa', '.txt']}
                maxSize={10 * 1024 * 1024} // 10MB
                disabled={isAnalyzing}
              />
            )}
          </motion.div>

          {/* Results Section */}
          {results.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="space-y-6"
            >
              <h2 className="text-2xl font-semibold text-gray-900">Analysis Results</h2>
              {results.map((result) => (
                <ResultsDisplay key={result.id} result={result} />
              ))}
            </motion.div>
          )}

          {/* Loading State */}
          {isAnalyzing && results.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-12"
            >
              <LoadingSpinner size="lg" />
              <p className="mt-4 text-gray-600">Analyzing sequence with TRAINED-PATHO model...</p>
              <p className="text-sm text-gray-500">This may take a few moments</p>
            </motion.div>
          )}
        </main>

        <Footer />
      </div>
    </>
  );
};

export default HomePage;