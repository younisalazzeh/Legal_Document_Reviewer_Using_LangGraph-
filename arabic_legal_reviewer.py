import os
import json
from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass
from pathlib import Path
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import PyPDF2
import docx
from io import BytesIO

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4"
    TEMPERATURE = 0.1

# State definition for LangGraph
class ReviewState(TypedDict):
    document_text: str
    document_type: str
    language: str  # 'arabic' or 'english'
    summary: str
    risk_flags: List[Dict[str, Any]]
    lawyer_questions: List[str]
    plain_english_clauses: List[Dict[str, str]]
    current_step: str
    error: str

@dataclass
class RiskFlag:
    severity: str  # "منخفض", "متوسط", "عالي", "حرج" for Arabic
    category: str
    description: str
    clause_text: str
    recommendation: str

class DocumentProcessor:
    """Handles document upload and text extraction with Arabic support"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect if text is primarily Arabic or English"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return "english"  # Default
        
        arabic_ratio = arabic_chars / total_chars
        return "arabic" if arabic_ratio > 0.5 else "english"
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        try:
            doc = docx.Document(BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(file_bytes: bytes) -> str:
        try:
            # Try UTF-8 first, then other encodings for Arabic
            encodings = ['utf-8', 'utf-16', 'cp1256', 'iso-8859-6']
            for encoding in encodings:
                try:
                    return file_bytes.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise Exception("Could not decode text with any supported encoding")
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")

class ArabicLegalReviewAgent:
    """Legal review agent with Arabic and English support"""
    
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ReviewState)
        
        # Add nodes
        workflow.add_node("summarize", self._summarize_document)
        workflow.add_node("flag_risks", self._flag_risks)
        workflow.add_node("generate_questions", self._generate_lawyer_questions)
        workflow.add_node("rewrite_clauses", self._rewrite_in_plain_language)
        
        # Define the flow
        workflow.set_entry_point("summarize")
        workflow.add_edge("summarize", "flag_risks")
        workflow.add_edge("flag_risks", "generate_questions")
        workflow.add_edge("generate_questions", "rewrite_clauses")
        workflow.add_edge("rewrite_clauses", END)
        
        return workflow.compile()
    
    def _get_prompts(self, language: str) -> Dict[str, str]:
        """Get language-specific prompts"""
        
        if language == "arabic":
            return {
                "summarize": """
أنت خبير قانوني متخصص في تلخيص الوثائق القانونية باللغة العربية.

نص الوثيقة:
{document_text}

يرجى تقديم ملخص شامل يتضمن:
1. نوع الوثيقة والغرض منها
2. الأطراف المشاركة
3. الشروط والأحكام الرئيسية
4. التواريخ والمواعيد النهائية المهمة
5. الالتزامات المالية
6. الحقوق والمسؤوليات الأساسية

اجعل الملخص واضحاً ومختصراً ومركزاً على أهم الجوانب القانونية.
""",
                
                "risks": """
أنت خبير في تقييم المخاطر القانونية. حلل الوثيقة التالية لتحديد المخاطر المحتملة والعلامات الحمراء.

نص الوثيقة:
{document_text}

لكل مخاطرة تحددها، قدم:
1. مستوى الخطورة (حرج، عالي، متوسط، منخفض)
2. فئة المخاطرة (مالية، مسؤولية، امتثال، إنهاء، إلخ)
3. وصف المخاطرة
4. النص أو البند المحدد الذي يحتوي على المخاطرة
5. التوصية لمعالجة المخاطرة

قم بتنسيق إجابتك كمصفوفة JSON:
[
    {{
        "severity": "عالي",
        "category": "مالية",
        "description": "تعرض غير محدود للمسؤولية",
        "clause_text": "نص البند المحدد هنا",
        "recommendation": "يُنصح بإضافة حد أقصى للمسؤولية"
    }}
]

أرجع فقط مصفوفة JSON، بدون نص آخر.
""",
                
                "questions": """
بناءً على الوثيقة القانونية والمخاطر المحددة، قم بإنشاء أسئلة مهمة يجب طرحها على المحامي.

ملخص الوثيقة:
{summary}

المخاطر المحددة:
{risks}

أنشئ 8-12 سؤالاً محدداً وقابلاً للتنفيذ يجب على العميل طرحها على محاميه حول هذه الوثيقة.
ركز على:
1. توضيح المصطلحات الغامضة
2. فهم تداعيات البنود المحفوفة بالمخاطر
3. الفرص التفاوضية
4. متطلبات الامتثال
5. استراتيجيات الخروج وإجراءات الإنهاء

قم بالتنسيق كمصفوفة JSON من النصوص:
["السؤال الأول؟", "السؤال الثاني؟", ...]

أرجع فقط مصفوفة JSON، بدون نص آخر.
""",
                
                "rewrite": """
مهمتك هي إعادة كتابة البنود القانونية المعقدة بلغة بسيطة ومفهومة لغير المختصين في القانون.

نص الوثيقة:
{document_text}

حدد 5-8 من أكثر البنود تعقيداً أو أهمية في هذه الوثيقة وأعد كتابتها بلغة بسيطة.
لكل بند:
1. استخرج النص القانوني المعقد الأصلي
2. قدم شرحاً بلغة بسيطة
3. اشرح ما يعنيه هذا من الناحية العملية

قم بالتنسيق كمصفوفة JSON:
[
    {{
        "original_clause": "النص القانوني الأصلي هنا",
        "plain_language": "الشرح البسيط هنا",
        "practical_meaning": "ما يعنيه هذا من الناحية العملية"
    }}
]

أرجع فقط مصفوفة JSON، بدون نص آخر.
"""
            }
        else:  # English prompts (original)
            return {
                "summarize": """
You are a legal expert tasked with summarizing a legal document. 

Document Text:
{document_text}

Please provide a comprehensive summary that includes:
1. Document type and purpose
2. Key parties involved
3. Main terms and conditions
4. Important dates and deadlines
5. Financial obligations
6. Key rights and responsibilities

Keep the summary clear, concise, and focused on the most important legal aspects.
""",
                
                "risks": """
You are a legal risk assessment expert. Analyze the following document for potential risks and red flags.

Document Text:
{document_text}

For each risk you identify, provide:
1. Severity level (critical, high, medium, low)
2. Risk category (e.g., financial, liability, compliance, termination, etc.)
3. Description of the risk
4. Specific clause or text that contains the risk
5. Recommendation for addressing the risk

Format your response as a JSON array of risk objects:
[
    {{
        "severity": "high",
        "category": "financial",
        "description": "Unlimited liability exposure",
        "clause_text": "specific clause text here",
        "recommendation": "recommend adding liability cap"
    }}
]

Only return the JSON array, no other text.
""",
                
                "questions": """
Based on the legal document and identified risks, generate important questions that should be asked to a lawyer.

Document Summary:
{summary}

Identified Risks:
{risks}

Generate 8-12 specific, actionable questions that a client should ask their lawyer about this document. 
Focus on:
1. Clarifying ambiguous terms
2. Understanding implications of risky clauses
3. Negotiation opportunities
4. Compliance requirements
5. Exit strategies and termination procedures

Format as a JSON array of strings:
["Question 1?", "Question 2?", ...]

Only return the JSON array, no other text.
""",
                
                "rewrite": """
You are tasked with rewriting complex legal clauses in plain language to make them understandable to non-lawyers.

Document Text:
{document_text}

Identify the 5-8 most complex or important clauses in this document and rewrite them in plain language.
For each clause:
1. Extract the original complex text
2. Provide a plain language explanation
3. Highlight what this means in practical terms

Format as a JSON array:
[
    {{
        "original_clause": "original legal text here",
        "plain_language": "simple explanation here",
        "practical_meaning": "what this means in real terms"
    }}
]

Only return the JSON array, no other text.
"""
            }
    
    def _summarize_document(self, state: ReviewState) -> ReviewState:
        """Step 1: Summarize the document terms"""
        prompts = self._get_prompts(state["language"])
        prompt_template = ChatPromptTemplate.from_template(prompts["summarize"])
        
        try:
            response = self.llm.invoke(prompt_template.format(document_text=state["document_text"]))
            state["summary"] = response.content
            state["current_step"] = "summarize_complete"
        except Exception as e:
            state["error"] = f"Error in summarization: {str(e)}"
        
        return state
    
    def _flag_risks(self, state: ReviewState) -> ReviewState:
        """Step 2: Flag potential risks"""
        prompts = self._get_prompts(state["language"])
        prompt_template = ChatPromptTemplate.from_template(prompts["risks"])
        
        try:
            response = self.llm.invoke(prompt_template.format(document_text=state["document_text"]))
            risk_flags = json.loads(response.content)
            state["risk_flags"] = risk_flags
            state["current_step"] = "risk_analysis_complete"
        except Exception as e:
            state["error"] = f"Error in risk analysis: {str(e)}"
            state["risk_flags"] = []
        
        return state
    
    def _generate_lawyer_questions(self, state: ReviewState) -> ReviewState:
        """Step 3: Generate questions for a lawyer"""
        prompts = self._get_prompts(state["language"])
        prompt_template = ChatPromptTemplate.from_template(prompts["questions"])
        
        try:
            risks_text = json.dumps(state["risk_flags"], indent=2, ensure_ascii=False)
            response = self.llm.invoke(prompt_template.format(
                summary=state["summary"],
                risks=risks_text
            ))
            questions = json.loads(response.content)
            state["lawyer_questions"] = questions
            state["current_step"] = "questions_generated"
        except Exception as e:
            state["error"] = f"Error generating questions: {str(e)}"
            state["lawyer_questions"] = []
        
        return state
    
    def _rewrite_in_plain_language(self, state: ReviewState) -> ReviewState:
        """Step 4: Rewrite complex clauses in plain language"""
        prompts = self._get_prompts(state["language"])
        prompt_template = ChatPromptTemplate.from_template(prompts["rewrite"])
        
        try:
            response = self.llm.invoke(prompt_template.format(document_text=state["document_text"]))
            clauses = json.loads(response.content)
            state["plain_english_clauses"] = clauses
            state["current_step"] = "rewriting_complete"
        except Exception as e:
            state["error"] = f"Error rewriting clauses: {str(e)}"
            state["plain_english_clauses"] = []
        
        return state
    
    def review_document(self, document_text: str, document_type: str = "contract") -> ReviewState:
        """Main method to review a document"""
        
        # Detect language
        language = DocumentProcessor.detect_language(document_text)
        
        initial_state = ReviewState(
            document_text=document_text,
            document_type=document_type,
            language=language,
            summary="",
            risk_flags=[],
            lawyer_questions=[],
            plain_english_clauses=[],
            current_step="starting",
            error=""
        )
        
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            initial_state["error"] = f"Workflow error: {str(e)}"
            return initial_state

def main():
    """Streamlit UI with Arabic support"""
    
    st.set_page_config(
        page_title="مراجع الوثائق القانونية / Legal Document Reviewer",
        page_icon="⚖️",
        layout="wide"
    )
    
    # Language selection
    language_option = st.selectbox(
        "اختر اللغة / Select Language",
        ["العربية / Arabic", "English / الإنجليزية"]
    )
    
    is_arabic_ui = language_option.startswith("العربية")
    
    # Apply RTL CSS for Arabic
    if is_arabic_ui:
        st.markdown("""
        <style>
        .main .block-container {
            direction: rtl;
            text-align: right;
        }
        .stSelectbox label {
            direction: rtl;
        }
        .stTextArea label {
            direction: rtl;
        }
        .stFileUploader label {
            direction: rtl;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Title and description
    if is_arabic_ui:
        st.title("⚖️ مراجع الوثائق القانونية")
        st.markdown("ارفع وثيقة قانونية للحصول على تحليل شامل مدعوم بالذكاء الاصطناعي")
    else:
        st.title("⚖️ Legal Document Reviewer")
        st.markdown("Upload a legal document for comprehensive AI-powered analysis")
    
    # Sidebar
    with st.sidebar:
        if is_arabic_ui:
            st.header("الإعدادات")
            api_key = st.text_input("مفتاح OpenAI API", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            st.markdown("---")
            st.markdown("### الصيغ المدعومة")
            st.markdown("- PDF (.pdf)")
            st.markdown("- Word (.docx)")
            st.markdown("- نص (.txt)")
        else:
            st.header("Configuration")
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            st.markdown("---")
            st.markdown("### Supported Formats")
            st.markdown("- PDF (.pdf)")
            st.markdown("- Word (.docx)")
            st.markdown("- Text (.txt)")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    # File upload
    upload_label = "اختر وثيقة قانونية" if is_arabic_ui else "Choose a legal document"
    upload_help = "ارفع عقد أو اتفاقية أو وثيقة قانونية أخرى للتحليل" if is_arabic_ui else "Upload a contract, agreement, or other legal document for analysis"
    
    uploaded_file = st.file_uploader(
        upload_label,
        type=['pdf', 'docx', 'txt'],
        help=upload_help
    )
    
    if uploaded_file is not None:
        if not api_key:
            error_msg = "يرجى إدخال مفتاح OpenAI API في الشريط الجانبي للمتابعة." if is_arabic_ui else "Please enter your OpenAI API key in the sidebar to proceed."
            st.error(error_msg)
            return
        
        try:
            # Extract text from uploaded file
            file_bytes = uploaded_file.read()
            
            if uploaded_file.type == "application/pdf":
                document_text = DocumentProcessor.extract_text_from_pdf(file_bytes)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                document_text = DocumentProcessor.extract_text_from_docx(file_bytes)
            elif uploaded_file.type == "text/plain":
                document_text = DocumentProcessor.extract_text_from_txt(file_bytes)
            else:
                error_msg = "نوع ملف غير مدعوم" if is_arabic_ui else "Unsupported file type"
                st.error(error_msg)
                return
            
            if len(document_text.strip()) == 0:
                error_msg = "لم يتم العثور على أي نص في الوثيقة" if is_arabic_ui else "No text could be extracted from the document"
                st.error(error_msg)
                return
            
            # Detect document language
            detected_language = DocumentProcessor.detect_language(document_text)
            lang_display = "العربية" if detected_language == "arabic" else "الإنجليزية"
            lang_display_en = "Arabic" if detected_language == "arabic" else "English"
            
            if is_arabic_ui:
                st.info(f"🔍 اللغة المكتشفة: {lang_display}")
            else:
                st.info(f"🔍 Detected Language: {lang_display_en}")
            
            # Show document preview
            preview_title = "📄 معاينة الوثيقة" if is_arabic_ui else "📄 Document Preview"
            with st.expander(preview_title):
                preview_text = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
                st.text_area("النص المستخرج" if is_arabic_ui else "Extracted Text", preview_text, height=200)
            
            # Process document
            button_text = "🔍 بدء التحليل" if is_arabic_ui else "🔍 Start Analysis"
            
            if st.button(button_text, type="primary"):
                progress_text = "جاري تحليل الوثيقة... قد يستغرق هذا بضع دقائق." if is_arabic_ui else "Analyzing document... This may take a few minutes."
                
                with st.spinner(progress_text):
                    try:
                        agent = ArabicLegalReviewAgent()
                        result = agent.review_document(document_text)
                        
                        if result["error"]:
                            error_msg = f"فشل التحليل: {result['error']}" if is_arabic_ui else f"Analysis failed: {result['error']}"
                            st.error(error_msg)
                            return
                        
                        # Display results
                        success_msg = "✅ اكتمل التحليل!" if is_arabic_ui else "✅ Analysis Complete!"
                        st.success(success_msg)
                        
                        # Summary Section
                        summary_title = "📋 ملخص الوثيقة" if is_arabic_ui else "📋 Document Summary"
                        st.header(summary_title)
                        st.markdown(result["summary"])
                        
                        # Risk Flags Section
                        risk_title = "⚠️ تحليل المخاطر" if is_arabic_ui else "⚠️ Risk Analysis"
                        st.header(risk_title)
                        
                        if result["risk_flags"]:
                            # Arabic severity mapping
                            severity_mapping = {
                                "حرج": "🔴", "critical": "🔴",
                                "عالي": "🟠", "high": "🟠",
                                "متوسط": "🟡", "medium": "🟡",  
                                "منخفض": "🟢", "low": "🟢"
                            }
                            
                            for i, risk in enumerate(result["risk_flags"]):
                                severity_icon = severity_mapping.get(risk["severity"], "⚪")
                                
                                if is_arabic_ui:
                                    title = f"{severity_icon} مخاطرة {risk['category']} - مستوى {risk['severity']}"
                                    desc_label = "**الوصف:**"
                                    clause_label = "**البند ذو الصلة:**"
                                    rec_label = "**التوصية:**"
                                else:
                                    title = f"{severity_icon} {risk['category'].title()} Risk - {risk['severity'].title()} Severity"
                                    desc_label = "**Description:**"
                                    clause_label = "**Relevant Clause:**"
                                    rec_label = "**Recommendation:**"
                                
                                with st.expander(title):
                                    st.markdown(f"{desc_label} {risk['description']}")
                                    st.markdown(f"{clause_label} {risk['clause_text']}")
                                    st.markdown(f"{rec_label} {risk['recommendation']}")
                        else:
                            no_risks_msg = "لم يتم تحديد مخاطر كبيرة" if is_arabic_ui else "No significant risks identified"
                            st.info(no_risks_msg)
                        
                        # Lawyer Questions Section
                        questions_title = "❓ أسئلة لمحاميك" if is_arabic_ui else "❓ Questions for Your Lawyer"
                        st.header(questions_title)
                        
                        if result["lawyer_questions"]:
                            for i, question in enumerate(result["lawyer_questions"], 1):
                                st.markdown(f"{i}. {question}")
                        else:
                            no_questions_msg = "لم يتم إنشاء أسئلة محددة" if is_arabic_ui else "No specific questions generated"
                            st.info(no_questions_msg)
                        
                        # Plain Language Section
                        plain_title = "📝 البنود المعقدة بلغة بسيطة" if is_arabic_ui else "📝 Complex Clauses in Plain Language"
                        st.header(plain_title)
                        
                        if result["plain_english_clauses"]:
                            for i, clause in enumerate(result["plain_english_clauses"]):
                                clause_title = f"البند {i+1}" if is_arabic_ui else f"Clause {i+1}"
                                
                                with st.expander(clause_title):
                                    if is_arabic_ui:
                                        st.markdown("**النص القانوني الأصلي:**")
                                        st.code(clause["original_clause"])
                                        st.markdown("**باللغة البسيطة:**")
                                        st.markdown(clause["plain_language"])
                                        st.markdown("**المعنى العملي:**")
                                        st.info(clause["practical_meaning"])
                                    else:
                                        st.markdown("**Original Legal Text:**")
                                        st.code(clause["original_clause"])
                                        st.markdown("**Plain Language:**")
                                        st.markdown(clause["plain_language"])
                                        st.markdown("**Practical Meaning:**")
                                        st.info(clause["practical_meaning"])
                        else:
                            no_clauses_msg = "لم يتم تحديد بنود معقدة للتبسيط" if is_arabic_ui else "No complex clauses identified for simplification"
                            st.info(no_clauses_msg)
                        
                        # Export functionality
                        export_title = "💾 تصدير النتائج" if is_arabic_ui else "💾 Export Results"
                        st.header(export_title)
                        
                        export_data = {
                            "document_language": result["language"],
                            "document_summary": result["summary"],
                            "risk_flags": result["risk_flags"],
                            "lawyer_questions": result["lawyer_questions"],
                            "plain_language_clauses": result["plain_english_clauses"]
                        }
                        
                        download_label = "📥 تحميل تقرير التحليل (JSON)" if is_arabic_ui else "📥 Download Analysis Report (JSON)"
                        
                        st.download_button(
                            label=download_label,
                            data=json.dumps(export_data, indent=2, ensure_ascii=False),
                            file_name=f"legal_analysis_{uploaded_file.name}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        error_msg = f"حدث خطأ أثناء التحليل: {str(e)}" if is_arabic_ui else f"An error occurred during analysis: {str(e)}"
                        st.error(error_msg)
                        
        except Exception as e:
            error_msg = f"خطأ في معالجة الملف: {str(e)}" if is_arabic_ui else f"Error processing file: {str(e)}"
            st.error(error_msg)

if __name__ == "__main__":
    main()