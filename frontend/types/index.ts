export interface PredictionRequest {
  sequence: string;
  format?: 'fasta' | 'fastq' | 'text';
  model?: string;
  threshold?: number;
}

export interface PredictionResponse {
  pathogenicity_score: number;
  pathogenicity_type: PathogenicityClass;
  confidence: number;
  sequence_info: {
    length: number;
    gc_content: number;
    nucleotide_counts?: Record<string, number>;
  };
  detailed_results?: {
    [key: string]: number | string;
    prob_benign?: number;
    prob_likely_benign?: number;
    prob_uncertain_significance?: number;
    prob_likely_pathogenic?: number;
    prob_pathogenic?: number;
    top_features?: Array<{
      feature: string;
      importance: number;
    }>;
  };
  model_info: {
    name: string;
    version: string;
    description?: string;
  };
}

export type PathogenicityClass = 
  | 'benign'
  | 'likely_benign'
  | 'uncertain_significance'
  | 'likely_pathogenic'
  | 'pathogenic';

export interface AnalysisResult {
  id: string;
  sequence: string;
  prediction: PredictionResponse;
  timestamp: Date;
  inputMethod: 'text' | 'file';
  fileName?: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  description?: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  description: string;
  features: number;
  training_samples: number;
  accuracy?: number;
}

export interface ApiError {
  message: string;
  details?: string;
  code?: string;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedFormats: string[];
  maxSize: number;
  disabled?: boolean;
}

export interface SequenceInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  maxLength?: number;
}

export interface ResultsDisplayProps {
  result: AnalysisResult;
  showDetails?: boolean;
  onToggleDetails?: () => void;
}