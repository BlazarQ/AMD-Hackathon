# Hackathon Agent

## Quick Start

1. Clone or download this repository
2. Run:
   ```bash
   docker compose up
   ```
3. Check the `output/results.json` file for results

## Using with API Keys

Create a `.env` file in the same directory with:
```
FIREWORKS_API_KEY=your-key-here
FIREWORKS_BASE_URL=your-url-here
ALLOWED_MODELS=your-model-here
```

Then run:
```bash
docker compose up
```

## Manual Docker Command

If not using docker-compose:
```bash
docker run -v output:/output ryqaz/agent:latest
```

Then view results:
```bash
cat output/results.json
```
