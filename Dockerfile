# Build stage
FROM rust:latest AS builder

WORKDIR /app

# Install dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifest files
COPY Cargo.toml ./

COPY src ./src

# Build the project in release mode
RUN cargo build --release

COPY Cargo.lock ./

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/target/release/burn-mnist /app/burn-mnist

# Make the binary executable
RUN chmod +x /app/burn-mnist

# Set environment variables for short training
ENV NUM_EPOCHS=2
ENV BATCH_SIZE=128
ENV NUM_WORKERS=2

# Run the training
CMD ["./burn-mnist"]

