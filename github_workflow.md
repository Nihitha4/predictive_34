# GitHub Workflow Guide

## 1. Create a repository

- Log in to GitHub.
- Click "New repository".
- Name the repo, choose public/private, and initialize with a README if needed.
- Copy the repository URL.

## 2. Create a local branch

```bash
git checkout -b feature/ml-projects
```

## 3. Add and commit your code

```bash
git add .
git commit -m "Add predictive ML project files"
```

## 4. Push to GitHub

```bash
git push -u origin feature/ml-projects
```

## 5. Open a pull request

- Go to the repository on GitHub.
- Click "Compare & pull request".
- Add a short summary and request review.

## 6. Merge the branch

- Merge after review.
- Then update your local main branch:

```bash
git checkout main
git pull origin main
```

This repository is prepared for those basic operations and for uploading predictive ML project work.
