import Link from 'next/link';
import { BeakerIcon, HeartIcon } from '@heroicons/react/24/outline';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white border-t border-gray-200 mt-16">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <BeakerIcon className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">
                Genome <span className="text-primary-600">AI</span>
              </span>
            </div>
            <p className="text-gray-600 max-w-md">
              Advanced AI-powered genome pathogenicity prediction using the TRAINED-PATHO model, 
              trained on expert-curated ClinVar variants for accurate clinical genomics analysis.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Resources</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/about" className="text-gray-600 hover:text-primary-600 transition-colors duration-200">
                  About
                </Link>
              </li>
              <li>
                <Link href="/docs" className="text-gray-600 hover:text-primary-600 transition-colors duration-200">
                  Documentation
                </Link>
              </li>
              <li>
                <a 
                  href="https://www.ncbi.nlm.nih.gov/clinvar/" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-600 hover:text-primary-600 transition-colors duration-200"
                >
                  ClinVar Database
                </a>
              </li>
              <li>
                <a 
                  href="https://github.com/yourusername/genome-pathogenicity-ai" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-600 hover:text-primary-600 transition-colors duration-200"
                >
                  GitHub Repository
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Connect</h3>
            <ul className="space-y-2">
              <li>
                <a 
                  href="mailto:contact@genomeai.com" 
                  className="text-gray-600 hover:text-primary-600 transition-colors duration-200"
                >
                  Contact Us
                </a>
              </li>
              <li>
                <a 
                  href="https://twitter.com/genomeai" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-600 hover:text-primary-600 transition-colors duration-200"
                >
                  Twitter
                </a>
              </li>
              <li>
                <a 
                  href="https://linkedin.com/company/genomeai" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-600 hover:text-primary-600 transition-colors duration-200"
                >
                  LinkedIn
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-gray-200 mt-8 pt-8 flex flex-col md:flex-row items-center justify-between">
          <div className="text-gray-600 text-sm">
            © {currentYear} Genome Pathogenicity AI. Built for research and educational purposes.
          </div>
          
          <div className="flex items-center text-sm text-gray-600 mt-4 md:mt-0">
            Made with <HeartIcon className="h-4 w-4 text-red-500 mx-1" /> for genomic research
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <p className="text-xs text-gray-500 text-center">
            <strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
            Results should not be used for clinical decision-making without proper validation 
            and consultation with qualified healthcare professionals.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;