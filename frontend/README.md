# Genome Pathogenicity AI - Frontend

A modern React/Next.js frontend for the Genome Pathogenicity AI Predictor, featuring the TRAINED-PATHO model interface.

## 🚀 Live Demo

Visit the live application: [https://genome-pathogenicity-ai.vercel.app](https://genome-pathogenicity-ai.vercel.app)

## ✨ Features

- **Interactive Sequence Analysis**: Text input and file upload for genome sequences
- **Real-time Predictions**: AI-powered pathogenicity scoring with confidence metrics
- **Detailed Results**: Class probabilities and feature importance analysis
- **Responsive Design**: Mobile-friendly interface with modern UI components
- **Model Information**: Comprehensive details about the TRAINED-PATHO model
- **Example Sequences**: Pre-loaded examples for quick testing

## 🛠️ Tech Stack

- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS with custom design system
- **Animation**: Framer Motion for smooth interactions
- **Icons**: Heroicons for consistent iconography
- **API Client**: Axios with mock API support
- **Deployment**: Vercel with GitHub Actions CI/CD

## 📋 Prerequisites

- Node.js 18 or higher
- npm or yarn package manager

## 🏃‍♂️ Quick Start

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/genome-pathogenicity-ai.git
cd genome-pathogenicity-ai/frontend

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

### Environment Configuration

Create a `.env.local` file with the following variables:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCK_API=true

# App Configuration
NEXT_PUBLIC_APP_NAME="Genome Pathogenicity AI"
NEXT_PUBLIC_APP_VERSION="1.0.0"
```

## 🚀 Deployment

### Deploy to Vercel (Recommended)

1. **Fork the repository** on GitHub
2. **Connect to Vercel**: 
   - Sign up at [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Set the root directory to `frontend`
3. **Configure environment variables** in Vercel dashboard:
   ```
   NEXT_PUBLIC_USE_MOCK_API=true
   NEXT_PUBLIC_API_URL=https://your-api-endpoint.com
   ```
4. **Deploy**: Vercel will automatically deploy on every push to main branch

### Manual Deployment

```bash
# Build the application
npm run build

# Export static files (optional)
npm run export

# Start production server
npm start
```

## 🔧 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run export` - Export static files

## 📁 Project Structure

```
frontend/
├── components/          # React components
│   ├── Header.tsx      # Navigation header
│   ├── Footer.tsx      # Site footer
│   ├── SequenceInput.tsx   # Sequence input component
│   ├── FileUpload.tsx      # File upload component
│   ├── ResultsDisplay.tsx  # Results visualization
│   └── ModelInfo.tsx       # Model information display
├── pages/              # Next.js pages
│   ├── _app.tsx       # App configuration
│   └── index.tsx      # Homepage
├── lib/               # Utilities and API
│   └── api.ts        # API service layer
├── types/             # TypeScript definitions
│   └── index.ts      # Type definitions
├── styles/            # CSS styles
│   └── globals.css   # Global styles and Tailwind
└── public/            # Static assets
```

## 🎨 Design System

The frontend uses a custom design system built on Tailwind CSS:

- **Colors**: Primary (blue), secondary (green), warning (yellow), danger (red)
- **Typography**: Inter font family with consistent sizing
- **Components**: Reusable UI components with consistent styling
- **Animations**: Smooth transitions and micro-interactions

## 🔌 API Integration

The frontend supports both real API and mock API modes:

### Mock API Mode (Default)
- Simulates realistic API responses
- Perfect for development and demos
- No backend required

### Real API Mode
- Connects to your Python backend
- Set `NEXT_PUBLIC_USE_MOCK_API=false`
- Configure `NEXT_PUBLIC_API_URL` to your backend URL

## 🧪 Testing

```bash
# Run linting
npm run lint

# Add tests (when available)
npm test
```

## 🚀 Performance Optimizations

- **Static Generation**: Pre-built pages for optimal performance
- **Code Splitting**: Automatic code splitting with Next.js
- **Image Optimization**: Optimized images with Next.js Image component
- **Bundle Analysis**: Analyze bundle size with `npm run analyze`

## 🌐 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Email: contact@genomeai.com
- Documentation: [Link to docs]

## 🙏 Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Icons from [Heroicons](https://heroicons.com/)
- Animations with [Framer Motion](https://www.framer.com/motion/)
- Deployed on [Vercel](https://vercel.com/)