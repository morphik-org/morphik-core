# Publishing the @morphik/ui Package

This document provides instructions for publishing the @morphik/ui package to npm.

## Prerequisites

1. You must have an npm account
2. You must be added as a contributor to the @morphik organization
3. You must be logged in to npm via the CLI (`npm login`)

## Publishing Steps

1. Ensure all changes are committed to the repository

2. Update the version in package.json (follow semantic versioning)
   ```bash
   npm version patch  # For bug fixes
   npm version minor  # For new features
   npm version major  # For breaking changes
   ```

3. Build the package
   ```bash
   npm run build:package
   ```

4. Run a dry-run to check the package contents
   ```bash
   npm pack --dry-run
   ```

5. Publish the package
   ```bash
   npm publish --access public
   ```

6. Create a git tag for the release
   ```bash
   git tag v$(node -p "require('./package.json').version")
   git push origin v$(node -p "require('./package.json').version")
   ```

## Installing for Local Development

If you want to test the package locally before publishing, you can use npm link:

1. In the ui-component directory:
   ```bash
   npm link
   ```

2. In your project that uses the package:
   ```bash
   npm link @morphik/ui
   ```

Alternatively, you can install from a local directory:

```bash
npm install /path/to/morphik-core/ui-component
```

Or from a GitHub repository:

```bash
npm install github:morphik-org/morphik-core#subdirectory=ui-component
```
