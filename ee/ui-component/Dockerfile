# syntax=docker/dockerfile:1

# Build stage
FROM node:23.11.0-alpine AS builder


# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig*.json ./
COPY next.config.mjs ./
COPY postcss.config.mjs ./
COPY tailwind.config.ts ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Print Node.js version for debugging
RUN node --version && npm --version


# Build the application
RUN npm run build

# Production stage
FROM node:23.11.0-alpine AS production

WORKDIR /app

# Set environment variables
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy necessary files from builder
COPY --from=builder /app/next.config.mjs ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Expose port
EXPOSE 3000

# Set the command to run the application
CMD ["node", "server.js"] 