import axios, { AxiosResponse } from 'axios';
import { PredictionRequest, PredictionResponse, ModelInfo, ApiError } from '../types';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 60000, // 60 seconds for genome analysis
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use((config) => {
  console.log('Making API request:', config.method?.toUpperCase(), config.url);
  return config;
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export class ApiService {
  /**
   * Predict pathogenicity for a given sequence
   */
  static async predictPathogenicity(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      const response: AxiosResponse<PredictionResponse> = await api.post('/predict', request);
      return response.data;
    } catch (error: any) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Get model information
   */
  static async getModelInfo(modelName?: string): Promise<ModelInfo> {
    try {
      const params = modelName ? { model: modelName } : {};
      const response: AxiosResponse<ModelInfo> = await api.get('/model/info', { params });
      return response.data;
    } catch (error: any) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Upload and analyze file
   */
  static async analyzeFile(file: File, options?: { model?: string; format?: string }): Promise<PredictionResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (options?.model) formData.append('model', options.model);
      if (options?.format) formData.append('format', options.format);

      const response: AxiosResponse<PredictionResponse> = await api.post('/predict/file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error: any) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Get available models
   */
  static async getAvailableModels(): Promise<string[]> {
    try {
      const response: AxiosResponse<{ models: string[] }> = await api.get('/models');
      return response.data.models;
    } catch (error: any) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Health check
   */
  static async healthCheck(): Promise<{ status: string; version: string }> {
    try {
      const response: AxiosResponse<{ status: string; version: string }> = await api.get('/health');
      return response.data;
    } catch (error: any) {
      throw this.handleApiError(error);
    }
  }

  /**
   * Handle API errors consistently
   */
  private static handleApiError(error: any): ApiError {
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data;
      
      return {
        message: data?.message || `Server error (${status})`,
        details: data?.details || error.response.statusText,
        code: status.toString(),
      };
    } else if (error.request) {
      // Request made but no response received
      return {
        message: 'Unable to connect to the server',
        details: 'Please check your internet connection and try again',
        code: 'NETWORK_ERROR',
      };
    } else {
      // Something else happened
      return {
        message: 'An unexpected error occurred',
        details: error.message,
        code: 'UNKNOWN_ERROR',
      };
    }
  }
}

// Mock API for development/demo when backend is not available
export class MockApiService {
  /**
   * Mock prediction for demo purposes
   */
  static async predictPathogenicity(request: PredictionRequest): Promise<PredictionResponse> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    const sequence = request.sequence;
    const length = sequence.length;
    const gcContent = this.calculateGCContent(sequence);
    
    // Simple mock logic for demonstration
    let pathogenicity_score = 0.5;
    let pathogenicity_type: PredictionResponse['pathogenicity_type'] = 'uncertain_significance';
    
    if (sequence.includes('AAAA') || sequence.includes('TTTT')) {
      pathogenicity_score = 0.8;
      pathogenicity_type = 'pathogenic';
    } else if (sequence.includes('CCC') || sequence.includes('GGG')) {
      pathogenicity_score = 0.2;
      pathogenicity_type = 'benign';
    } else if (gcContent > 0.6) {
      pathogenicity_score = 0.3;
      pathogenicity_type = 'likely_benign';
    } else if (gcContent < 0.3) {
      pathogenicity_score = 0.7;
      pathogenicity_type = 'likely_pathogenic';
    }

    const confidence = 0.65 + Math.random() * 0.25; // Random confidence between 0.65-0.9

    return {
      pathogenicity_score,
      pathogenicity_type,
      confidence,
      sequence_info: {
        length,
        gc_content: gcContent,
        nucleotide_counts: this.countNucleotides(sequence),
      },
      detailed_results: {
        prob_benign: pathogenicity_type === 'benign' ? 0.8 : 0.1,
        prob_likely_benign: pathogenicity_type === 'likely_benign' ? 0.7 : 0.15,
        prob_uncertain_significance: pathogenicity_type === 'uncertain_significance' ? 0.6 : 0.2,
        prob_likely_pathogenic: pathogenicity_type === 'likely_pathogenic' ? 0.75 : 0.1,
        prob_pathogenic: pathogenicity_type === 'pathogenic' ? 0.85 : 0.05,
        top_features: [
          { feature: 'gc_content', importance: 0.12 },
          { feature: 'sequence_length', importance: 0.10 },
          { feature: 'complexity', importance: 0.09 },
          { feature: 'entropy', importance: 0.08 },
          { feature: 'kmer_3_ATG', importance: 0.07 },
        ],
      },
      model_info: {
        name: request.model || 'TRAINED-PATHO',
        version: '1.0.0',
        description: 'Demo mode - using mock predictions',
      },
    };
  }

  static async getModelInfo(): Promise<ModelInfo> {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      name: 'TRAINED-PATHO',
      version: '1.0.0',
      description: 'ClinVar-trained pathogenicity prediction model',
      features: 159,
      training_samples: 11429,
      accuracy: 0.739,
    };
  }

  private static calculateGCContent(sequence: string): number {
    const gcCount = (sequence.match(/[GC]/gi) || []).length;
    return sequence.length > 0 ? gcCount / sequence.length : 0;
  }

  private static countNucleotides(sequence: string): Record<string, number> {
    const counts = { A: 0, T: 0, C: 0, G: 0, N: 0 };
    for (const nucleotide of sequence.toUpperCase()) {
      if (nucleotide in counts) {
        counts[nucleotide as keyof typeof counts]++;
      }
    }
    return counts;
  }
}

// Export the appropriate service based on environment
const isDevelopment = process.env.NODE_ENV === 'development';
const useMockApi = process.env.NEXT_PUBLIC_USE_MOCK_API === 'true';

export default (isDevelopment && useMockApi) ? MockApiService : ApiService;