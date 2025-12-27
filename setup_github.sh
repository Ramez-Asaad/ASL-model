
#!/bin/bash
set -e

echo "Initializing Git Repository..."
git init

echo "Installing Git LFS..."
git lfs install

echo "Tracking large files with LFS..."
git lfs track "*.csv"
git lfs track "*.pth"
git lfs track "*.zip"
git lfs track "*.npy" # In case any slip through, though we ignored processed dirs

echo "Adding files..."
git add .

echo "Committing..."
git commit -m "Initial commit for ASL Service (IPNHand Integration)"

echo "---------------------------------------------------"
echo "Repository Setup Complete!"
echo "To push to GitHub:"
echo "1. Create a new repository on GitHub."
echo "2. Run: git remote add origin <YOUR_REPO_URL>"
echo "3. Run: git push -u origin main"
echo "---------------------------------------------------"
