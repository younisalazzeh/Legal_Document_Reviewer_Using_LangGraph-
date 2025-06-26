# ⚖️ Legal Document Reviewer

An AI-powered legal document analysis tool built with LangGraph that provides comprehensive contract review through a step-by-step workflow. Upload any legal document and get automated analysis including term summaries, risk assessment, lawyer questions, and plain English translations.

## 🚀 Features

- **Step-by-Step Analysis**: Uses LangGraph to orchestrate a structured review workflow
- **Document Summary**: Extracts key terms, parties, obligations, and dates
- **Risk Assessment**: Identifies and categorizes potential legal risks with severity levels
- **Lawyer Questions**: Generates specific questions to ask your legal counsel
- **Plain English Translation**: Rewrites complex legal clauses in understandable language
- **Multi-Format Support**: Works with PDF, DOCX, and TXT files
- **Export Functionality**: Download analysis results as structured JSON

## 📋 Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed
- An OpenAI API key (get one at https://platform.openai.com/api-keys)
- Basic familiarity with command line operations

## 🛠️ Installation

### Step 1: Clone or Download the Project

Create a new directory for your project and save the main application file as `legal_reviewer.py`.

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv legal_reviewer_env

# Activate virtual environment
# On Windows:
legal_reviewer_env\Scripts\activate
# On macOS/Linux:
source legal_reviewer_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in your project directory:

```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🏃‍♂️ Running the Application

### Start the Streamlit Application

```bash
streamlit run legal_reviewer.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Enter API Key**: If not set in environment variables, enter your OpenAI API key in the sidebar
2. **Upload Document**: Click "Choose a legal document" and select your PDF, DOCX, or TXT file
3. **Start Analysis**: Click the "🔍 Start Analysis" button
4. **Review Results**: The application will display:
   - Document summary
   - Risk analysis with severity levels
   - Questions for your lawyer
   - Complex clauses in plain English
5. **Export Results**: Download the analysis as a JSON file for future reference

## 🏗️ Project Architecture

### Core Components

#### 1. **LangGraph Workflow**
The application uses LangGraph to create a structured, step-by-step analysis workflow:

```
Document Upload → Summarize → Flag Risks → Generate Questions → Rewrite Clauses → Results
```

#### 2. **State Management**
```python
class ReviewState(TypedDict):
    document_text: str
    document_type: str
    summary: str
    risk_flags: List[Dict[str, Any]]
    lawyer_questions: List[str]
    plain_english_clauses: List[Dict[str, str]]
    current_step: str
    error: str
```

#### 3. **Document Processing**
- **PDF Processing**: Uses PyPDF2 for text extraction
- **DOCX Processing**: Uses python-docx for Word document parsing
- **TXT Processing**: Direct UTF-8 text reading

#### 4. **AI Agent System**
- **LLM Integration**: OpenAI GPT-4 for intelligent analysis
- **Prompt Engineering**: Specialized prompts for each analysis step
- **Error Handling**: Comprehensive error management throughout workflow

### Workflow Steps Explained

#### Step 1: Document Summarization
- **Purpose**: Extract key information and provide overview
- **Output**: Structured summary including parties, terms, dates, obligations
- **Prompt Strategy**: Uses template-based prompting for consistent results

#### Step 2: Risk Analysis
- **Purpose**: Identify potential legal risks and red flags
- **Output**: Categorized risks with severity levels and recommendations
- **Risk Categories**: Financial, liability, compliance, termination, etc.
- **Severity Levels**: Critical, High, Medium, Low

#### Step 3: Lawyer Questions Generation
- **Purpose**: Create actionable questions for legal consultation
- **Focus Areas**: Ambiguous terms, risky clauses, negotiation opportunities
- **Output**: 8-12 specific, targeted questions

#### Step 4: Plain English Rewriting
- **Purpose**: Make complex legal language accessible
- **Process**: Identifies complex clauses and provides simplified explanations
- **Output**: Original text, plain English version, practical meaning

## 🔧 Customization Options

### Modifying AI Model Settings

In the `Config` class, you can adjust:
```python
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4"  # Change to gpt-3.5-turbo for faster/cheaper analysis
    TEMPERATURE = 0.1     # Adjust creativity vs consistency
```

### Adding New Analysis Steps

To add a new step to the workflow:

1. **Create the analysis function**:
```python
def _new_analysis_step(self, state: ReviewState) -> ReviewState:
    # Your analysis logic here
    return state
```

2. **Update the graph**:
```python
workflow.add_node("new_step", self._new_analysis_step)
workflow.add_edge("previous_step", "new_step")
```

3. **Update the state**:
```python
class ReviewState(TypedDict):
    # Add new fields for your analysis
    new_field: List[str]
```

### Customizing Risk Categories

Modify the risk flagging prompt to focus on specific areas:
- Add industry-specific risks
- Customize severity thresholds
- Include regulatory compliance checks

## 🐛 Troubleshooting

### Common Issues

#### "OpenAI API key not found"
- Ensure your API key is correctly set in the `.env` file
- Check that the environment variable is loaded
- Verify your API key is valid and has sufficient credits

#### "Error reading PDF/DOCX"
- Ensure the file is not password-protected
- Try converting to a different format
- Check file size (very large files may timeout)

#### "Analysis taking too long"
- Large documents may take several minutes
- Consider using gpt-3.5-turbo for faster processing
- Break very large documents into sections

#### "JSON parsing errors"
- The AI occasionally returns malformed JSON
- The application includes error handling for this
- Retry the analysis if errors persist

### Performance Tips

1. **Document Size**: Optimal performance with documents under 50 pages
2. **Model Selection**: Use GPT-3.5-turbo for faster processing, GPT-4 for better accuracy
3. **API Rate Limits**: Be aware of OpenAI's rate limiting for your tier

## 📊 Output Format

### Risk Analysis Output
```json
{
  "severity": "high",
  "category": "financial",
  "description": "Unlimited liability exposure",
  "clause_text": "Party A shall be liable for all damages...",
  "recommendation": "Add liability cap clause"
}
```

### Plain English Clauses Output
```json
{
  "original_clause": "Complex legal text...",
  "plain_english": "Simple explanation...",
  "practical_meaning": "What this means for you..."
}
```

## 🔐 Security and Privacy

- **API Key Security**: Never commit API keys to version control
- **Document Privacy**: Documents are processed via OpenAI's API (review their privacy policy)
- **Local Processing**: Consider using local LLMs for sensitive documents
- **Data Retention**: No document data is stored by this application

## 🚀 Deployment Options

### Local Deployment
Run directly on your machine using the installation steps above.

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Add OpenAI API key to secrets

#### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "legal_reviewer.py"]
```

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

### Areas for Enhancement

- **Additional Document Formats**: Support for more file types
- **Multiple Language Support**: Analysis in different languages
- **Batch Processing**: Handle multiple documents simultaneously
- **Integration APIs**: REST API for programmatic access
- **Advanced Risk Scoring**: ML-based risk assessment
- **Template Library**: Pre-built analysis templates for different contract types

## 📄 License

This project is open source. Please ensure compliance with OpenAI's usage policies when using their API.

## 🆘 Support

For support:
1. Check the troubleshooting section above
2. Review OpenAI API documentation for API-related issues
3. Create an issue in the project repository with detailed error information

## 🔮 Future Enhancements

- **Clause Database**: Build a knowledge base of common legal clauses
- **Comparison Tool**: Compare multiple contract versions
- **Automated Suggestions**: AI-powered clause recommendations
- **Integration**: Connect with legal practice management systems
- **Mobile App**: Native mobile application for document review
- **Voice Interface**: Voice-activated document queries

---

**Disclaimer**: This tool is for informational purposes only and does not constitute legal advice. Always consult with qualified legal professionals for important legal matters.
