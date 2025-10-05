# Wikimedia Curation Agent

An intelligent image curation system that uses LangGraph workflows and OpenAI Vision API to automatically find, analyze, and curate high-quality images from Wikimedia Commons for biographical content.

## Features

- **AI-Powered Curation**: Uses OpenAI's vision model to analyze images for portrait quality, historical significance, and technical quality
- **LangGraph Workflows**: Structured workflow with search, curation, and storage steps
- **Duplicate Detection**: Automatically removes similar images to ensure variety
- **MongoDB Integration**: Stores curated results with rich metadata
- **Two Image Types**: Supports both primary photos (of people) and context photos (historical background)
- **Rich Terminal Output**: Beautiful formatted summaries with scores, details, and MongoDB IDs

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/wikimedia-agent
cd wikimedia-agent
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit `.env` with your credentials:

```env
# MongoDB Configuration (required)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database

# OpenAI API Configuration (required)
OPENAI_API_KEY=your-openai-api-key

# Optional: Anthropic API for additional features
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Usage

### Run the Example
```bash
python src/agent.py
```

### Programmatic Usage
```python
import asyncio
from src.agent import run_curation_workflow

async def main():
    # For primary photos of a person
    result = await run_curation_workflow(
        subject="Marie Curie",
        is_primary_photos=True,  # Photos OF the person
        limit=8
    )
    
    # For context/background photos
    result = await run_curation_workflow(
        subject="Renaissance",
        is_primary_photos=False,  # Historical context photos
        limit=5
    )
    
    print(result)

asyncio.run(main())
```

### Run Tests
```bash
python tests/test_agent.py
```

## Project Structure

```
wikimedia-agent/
   src/                    # Main source code
      agent.py           # Main LangGraph agent
      tools/             # External API tools
         wikimedia.py   # Wikimedia Commons integration
      utils/             # Utilities
          mongodb_helper.py
          openai_helper.py
   workflows/             # AI curation workflows
      image_curator.py   # Image analysis and curation
   tests/                 # Test files
      test_agent.py      # Main test suite
   .env                   # Environment variables (gitignored)
   .env.example          # Template for environment setup
   requirements.txt      # Python dependencies
```

## How It Works

1. **Search**: Queries Wikimedia Commons for images related to the subject
2. **Filter**: Removes video files and invalid formats
3. **Deduplicate**: Identifies and removes similar images
4. **Analyze**: Uses OpenAI Vision API to score each image on:
   - Portrait quality (0-10)
   - Historical significance (0-10) 
   - Technical quality (0-10)
5. **Curate**: Selects best images and creates informative captions
6. **Store**: Saves results to MongoDB with rich metadata

## Output Format

Results include:
- **Featured images**: Highest-rated portraits (usually 1)
- **Carousel images**: Additional high-quality images
- **Metadata**: Dimensions, attribution, license info
- **AI scores**: Portrait, historical, and quality ratings
- **Curated captions**: Descriptive, specific captions

### Terminal Output Example

```
🚀 Starting Wikimedia curation for: Marie Curie
📷 Type: Primary photos
🔢 Limit: 3 images

================================================================================
🖼️  WIKIMEDIA CURATION RESULTS
================================================================================
✅ Status: SUCCESS
👤 Subject: Marie Curie
📷 Photo Type: Primary photos
🔍 Images Processed: 3
🎯 Images Curated: 3
💾 Images Saved: 3
💬 Message: Successfully saved 3 images to MongoDB

📋 CURATED IMAGES:
--------------------------------------------------------------------------------
⭐ FEATURED:
   📸 Marie Curie portrait, 1920s
      📊 Scores: Portrait=10/10, Historical=9/10, Quality=8/10
      📐 Size: 441×600
      🔗 Source: https://upload.wikimedia.org/wikipedia/commons/c/c8/Marie_Curie_c._1920s.jpg...

🎠 CAROUSEL:
   📸 Marie Curie portrait, pre-1907
      📊 Scores: Portrait=10/10, Historical=9/10, Quality=8/10
      📐 Size: 463×600
      🔗 Source: https://upload.wikimedia.org/wikipedia/commons/d/d9/Mariecurie.jpg...

🗄️  MongoDB Document IDs:
   1. 68b0d717b4a4e48f60a786c4
   2. 68b0d717b4a4e48f60a786c5
================================================================================
```

## API Reference

### `run_curation_workflow(subject, is_primary_photos, limit=8)`

**Parameters:**
- `subject` (str): Person or topic to search for
- `is_primary_photos` (bool): True for photos OF the subject, False for contextual photos
- `limit` (int): Maximum number of images to process

**Returns:**
- `success` (bool): Whether the workflow completed successfully
- `images_processed` (int): Number of candidate images found
- `images_curated` (int): Number of images after curation
- `images_saved` (int): Number of images saved to database
- `saved_image_ids` (list): MongoDB document IDs of saved images
- `message` (str): Success or error message

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Run `python tests/test_agent.py` to verify
6. Submit a pull request

## License

MIT License - see LICENSE file for details.