# LangBridge AI MVP

## Project Overview
LangBridge AI is a web-based Minimum Viable Product (MVP) that translates three Nigerian languages (Igbo, Yoruba, Hausa) to English in near real-time. The system accepts text or audio input, detects the input language, and displays English translations using AI/ML techniques. This tool aims to assist travelers, NGOs, and aid organizations.

**Key Features (MVP) **:
- Input: Text 
- Language Detection: Igbo, Yoruba, Hausa
- Translation: To English using HuggingFace models
- Web Interface: Simple UI built with Streamlit or Flask
- Tech Stack: PyTorch, HuggingFace, Streamlit/Flask, Pandas, NumPy, Git, Render/Railway

**Additional Features **:
- Audio input
- Audio feedback
  
## Setup Instructions
To set up the project locally:
1. Install Python 3.8+: `python --version`
2. Install dependencies:
   ```bash
   pip install torch transformers datasets flask streamlit pandas numpy
   ```
3. Clone the repository:
   ```bash
   git clone <repo-url>
   cd langbridge-ai
   ```
4. Run the app (after development):
   ```bash
   streamlit run src/app.py
   ```

## Project Plan
- **Days 1-2 (July 18-19, 2025)**: Set up environment, research HuggingFace models (e.g., MarianMT), datasets (JW300, Tatoeba), and draft plan.
- **Days 3-5 (July 20-22)**: Download and preprocess JW300/Tatoeba datasets using Pandas.
- **Days 6-8 (July 23-25)**: Fine-tune a HuggingFace model for Igbo/Yoruba/Hausa-to-English translation.
- **Days 9-10 (July 26-27)**: Build a Streamlit/Flask app for text input and translation output.
- **Days 11-13 (July 28-30)**: Integrate model with app, test, and polish UI.
- **Day 14 (July 31)**: Final testing, deploy to Render/Railway, record demo video, write report.

## Progress Updates
- **Day 1 (July 18, 2025)**: 
  - Started installing Python 3.8+, PyTorch, HuggingFace, Flask/Streamlit, Pandas, NumPy.
  - Created GitHub repository and initialized `README.md`.
  - (To be updated with research notes by end of Day 1)

## Research Notes
- **HuggingFace**: Provides pretrained models like MarianMT for translation (huggingface.co/docs/transformers/tasks/translation).
- **Datasets**:
  - JW300: Parallel text for African languages (jw300.org).
  - Tatoeba: Sentence pairs for translation (tatoeba.org).
  - AI4D: African language datasets (search GitHub for repositories).
- **Streamlit**: Simple framework for web apps (docs.streamlit.io).
- (To be updated with findings from Days 1-2 research)

## Challenges and Questions
- **Challenges**: 
  - Ensuring all libraries install correctly.
  - Understanding HuggingFace model fine-tuning for low-resource languages.
- **Questions**:
  - How to verify JW300 data contains Igbo/Yoruba/Hausa-to-English pairs?
  - What’s the easiest way to test a HuggingFace model locally?
- (To be updated with new challenges/questions daily)

## Resources
- HuggingFace Translation Tutorial: huggingface.co/docs/transformers/tasks/translation
- Streamlit Getting Started: docs.streamlit.io
- JW300 Dataset: jw300.org
- Tatoeba Dataset: tatoeba.org
- Git Tutorial: freecodecamp.org/news/learn-the-basics-of-git-in-under-10-minutes
- Render Deployment: render.com/docs

## License
MIT License (To be finalized based on internship requirements).
