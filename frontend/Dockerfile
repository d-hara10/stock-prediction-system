# Use official Node.js image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy rest of frontend code
COPY . .

# Build Next.js app (NO API KEY NEEDED)
RUN npm run build

# Expose port 3000
EXPOSE 3000

# Start Next.js in production mode
CMD ["npm", "start"]