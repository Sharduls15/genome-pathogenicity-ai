# 🚀 Deployment Guide - Genome Pathogenicity AI Frontend

This guide will help you deploy your Genome Pathogenicity AI frontend to GitHub and Vercel.

## 📋 Prerequisites

- GitHub account
- Vercel account
- Node.js 18+ installed locally
- Your trained TRAINED-PATHO model

## 🔧 Step 1: Prepare Your Repository

### 1.1 Create GitHub Repository

```bash
# Initialize git repository (if not already done)
cd C:\Users\shard\genome-pathogenicity-ai
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Genome Pathogenicity AI with TRAINED-PATHO model"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/genome-pathogenicity-ai.git

# Push to GitHub
git push -u origin main
```

### 1.2 Repository Structure
Ensure your repository has this structure:
```
genome-pathogenicity-ai/
├── frontend/               # React/Next.js frontend
│   ├── components/        # UI components
│   ├── pages/            # Next.js pages
│   ├── lib/              # API utilities
│   ├── types/            # TypeScript types
│   ├── styles/           # CSS styles
│   ├── package.json      # Dependencies
│   ├── next.config.js    # Next.js config
│   └── vercel.json       # Vercel config
├── src/                   # Python backend (optional)
├── models/               # ML model files
├── scripts/              # Training scripts
├── .github/workflows/    # GitHub Actions
└── README.md
```

## 🌐 Step 2: Deploy to Vercel

### 2.1 Connect to Vercel

1. **Sign up for Vercel**: Go to [vercel.com](https://vercel.com) and sign up with GitHub
2. **Import Repository**: 
   - Click "New Project"
   - Select your `genome-pathogenicity-ai` repository
   - Click "Import"

### 2.2 Configure Build Settings

In the Vercel import screen:

- **Framework Preset**: Next.js
- **Root Directory**: `frontend` ⚠️ **IMPORTANT**
- **Build Command**: `npm run build`
- **Output Directory**: `.next` (default)
- **Install Command**: `npm ci`

### 2.3 Environment Variables

Add these environment variables in Vercel dashboard:

```env
NEXT_PUBLIC_USE_MOCK_API=true
NEXT_PUBLIC_API_URL=https://your-backend-api.com
NEXT_PUBLIC_APP_NAME=Genome Pathogenicity AI
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### 2.4 Deploy

Click **"Deploy"** and wait for the build to complete (~2-3 minutes).

## 🔄 Step 3: Set Up Continuous Deployment

### 3.1 GitHub Actions (Already Configured)

Your repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that:
- Triggers on pushes to `main` branch
- Builds and tests the frontend
- Deploys to Vercel automatically

### 3.2 Configure GitHub Secrets

Add these secrets in your GitHub repository:

1. Go to **Settings > Secrets and variables > Actions**
2. Add these repository secrets:

```
VERCEL_TOKEN=your-vercel-token
VERCEL_ORG_ID=your-vercel-org-id  
VERCEL_PROJECT_ID=your-vercel-project-id
```

**To get these values:**
- **VERCEL_TOKEN**: Vercel Dashboard > Settings > Tokens > Create Token
- **VERCEL_ORG_ID**: Vercel Dashboard > Settings > General (Team ID)
- **VERCEL_PROJECT_ID**: Project Settings > General (Project ID)

## 🎯 Step 4: Configure Domain (Optional)

### 4.1 Custom Domain Setup

In Vercel dashboard:
1. Go to your project settings
2. Click "Domains" 
3. Add your custom domain
4. Configure DNS with your domain provider

### 4.2 SSL Certificate

Vercel automatically provides SSL certificates for all deployments.

## 🧪 Step 5: Test Your Deployment

### 5.1 Verify Frontend

1. Visit your Vercel URL (e.g., `https://genome-pathogenicity-ai.vercel.app`)
2. Test sequence input with example: `ATGCGATCGATCGATCGATCGATCGATC`
3. Test file upload with a FASTA file
4. Verify results display correctly

### 5.2 Check Mock API

Since `NEXT_PUBLIC_USE_MOCK_API=true`, the app uses simulated data:
- Predictions work without backend
- Demo mode displays realistic results
- Perfect for showcasing your model

## 🔗 Step 6: Connect Real Backend (Optional)

### 6.1 Deploy Python Backend

Deploy your Python backend with the TRAINED-PATHO model to:
- **Railway**: Easy Python deployment
- **Heroku**: Popular PaaS platform  
- **AWS/GCP/Azure**: Cloud platforms
- **Your own server**: VPS or dedicated server

### 6.2 Update Environment Variables

Once your backend is deployed:

```env
NEXT_PUBLIC_USE_MOCK_API=false
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

### 6.3 CORS Configuration

Ensure your Python backend allows requests from your Vercel domain:

```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins=['https://your-app.vercel.app'])
```

## 📊 Step 7: Monitoring & Analytics

### 7.1 Vercel Analytics

Enable Vercel Analytics in your project dashboard for:
- Page views and performance
- Core Web Vitals
- User engagement metrics

### 7.2 Error Monitoring

Consider adding error monitoring:
- **Sentry**: Error tracking and performance monitoring
- **LogRocket**: Session replay and debugging

## 🔄 Step 8: Updates & Maintenance

### 8.1 Automatic Deployments

Every push to `main` branch automatically:
1. Triggers GitHub Actions
2. Builds the application
3. Runs tests and linting
4. Deploys to Vercel

### 8.2 Manual Deployment

```bash
# Make changes to your code
git add .
git commit -m "Update: describe your changes"
git push origin main
# Vercel automatically deploys!
```

## 🛠️ Troubleshooting

### Common Issues

**Build Fails on Vercel:**
- Check build logs in Vercel dashboard
- Ensure `frontend` is set as root directory
- Verify all dependencies in `package.json`

**Environment Variables Not Working:**
- Check variable names (must start with `NEXT_PUBLIC_`)
- Redeploy after adding variables
- Check browser dev tools for values

**API Connection Issues:**
- Verify `NEXT_PUBLIC_API_URL` is correct
- Check CORS settings in backend
- Test API endpoints manually

**GitHub Actions Failing:**
- Check repository secrets are set correctly
- Verify Vercel token has correct permissions
- Review action logs for specific errors

### Performance Tips

1. **Optimize Images**: Use Next.js Image component
2. **Bundle Analysis**: Run `npm run build` and check bundle size
3. **Caching**: Vercel automatically caches static assets
4. **CDN**: Vercel uses global CDN for fast loading

## 📞 Support

If you encounter issues:

1. **Check Logs**: Vercel deployment logs and GitHub Actions logs
2. **GitHub Issues**: Open an issue in your repository
3. **Vercel Docs**: [vercel.com/docs](https://vercel.com/docs)
4. **Next.js Docs**: [nextjs.org/docs](https://nextjs.org/docs)

## 🎉 Success! 

Your Genome Pathogenicity AI frontend is now:

- ✅ Deployed to Vercel with custom domain
- ✅ Connected to GitHub for version control
- ✅ Automatically deploying on code changes
- ✅ Featuring your TRAINED-PATHO model
- ✅ Accessible worldwide with HTTPS
- ✅ Optimized for performance and SEO

**Live URL**: `https://your-app.vercel.app`

Share your deployed application and showcase your AI-powered genome pathogenicity prediction tool to the world! 🧬🚀