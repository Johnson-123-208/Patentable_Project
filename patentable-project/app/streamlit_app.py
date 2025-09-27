import streamlit as st
import requests
import json
import os
from typing import Dict, Any
import tempfile
from utils.export_report import generate_pdf_report
import io

# Configure page
st.set_page_config(
    page_title="Patent Discovery & Mentor Recommender",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .score-container {
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 1rem 0;
    }
    
    .patent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    .mentor-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    
    .high-score { color: #dc3545; }
    .medium-score { color: #fd7e14; }
    .low-score { color: #28a745; }
</style>
""", unsafe_allow_html=True)

def get_score_color(score: float) -> str:
    """Get color class based on patentability score"""
    if score >= 0.7:
        return "high-score"
    elif score >= 0.4:
        return "medium-score"
    else:
        return "low-score"

def get_score_interpretation(score: float) -> str:
    """Get human-readable interpretation of score"""
    if score >= 0.8:
        return "Very High - Strong patent potential"
    elif score >= 0.6:
        return "High - Good patent potential"
    elif score >= 0.4:
        return "Medium - Requires review"
    elif score >= 0.2:
        return "Low - Limited patent potential"
    else:
        return "Very Low - Unlikely to be patentable"

def call_backend_api(text: str, backend_url: str = "http://localhost:8000") -> Dict[Any, Any]:
    """Call the backend inference API"""
    try:
        response = requests.post(
            f"{backend_url}/infer",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend API Error: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid response format from backend")
        return None

def display_results(results: Dict[Any, Any]):
    """Display the inference results in a structured format"""
    
    # Patentability Score Section
    st.markdown("## 📊 Patentability Analysis")
    
    score = results.get('patentability_score', 0.0)
    score_color = get_score_color(score)
    score_text = get_score_interpretation(score)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="score-container">
            <h3>Patentability Score</h3>
            <h2 class="{score_color}">{score:.2f}</h2>
            <p><strong>{score_text}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Progress bar with color coding
        if score >= 0.7:
            bar_color = "#dc3545"  # Red
        elif score >= 0.4:
            bar_color = "#fd7e14"  # Orange
        else:
            bar_color = "#28a745"  # Green
            
        st.markdown("### Score Breakdown")
        st.progress(score)
        
        # Score components if available
        if 'score_components' in results:
            components = results['score_components']
            for component, value in components.items():
                st.metric(component.replace('_', ' ').title(), f"{value:.3f}")
    
    # Similar Patents Section
    st.markdown("## 📋 Similar Patents Found")
    
    similar_patents = results.get('similar_patents', [])
    if similar_patents:
        for i, patent in enumerate(similar_patents[:5], 1):
            similarity = patent.get('similarity', 0.0)
            st.markdown(f"""
            <div class="patent-card">
                <h4>#{i}. {patent.get('title', 'Patent Title Not Available')}</h4>
                <p><strong>Similarity:</strong> <span style="color: #1f4e79;">{similarity:.3f}</span></p>
                <p><strong>Patent ID:</strong> {patent.get('patent_id', 'N/A')}</p>
                <p><strong>Abstract:</strong> {patent.get('abstract', 'Abstract not available')[:200]}...</p>
                {f'<p><a href="{patent.get("url", "#")}" target="_blank">View Patent →</a></p>' if patent.get('url') else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No similar patents found in the database.")
    
    # Recommended Mentors Section
    st.markdown("## 👨‍🏫 Recommended Mentors")
    
    mentors = results.get('recommended_mentors', [])
    if mentors:
        cols = st.columns(min(3, len(mentors)))
        for i, mentor in enumerate(mentors[:3]):
            with cols[i]:
                st.markdown(f"""
                <div class="mentor-card">
                    <h4>{mentor.get('name', 'Mentor Name')}</h4>
                    <p><strong>Domain:</strong> {mentor.get('domain', 'Not specified')}</p>
                    <p><strong>Experience:</strong> {mentor.get('experience_years', 'N/A')} years</p>
                    <p><strong>Bio:</strong> {mentor.get('bio', 'Biography not available')[:150]}...</p>
                    <p><strong>Contact:</strong> {mentor.get('email', 'Not available')}</p>
                    <p><strong>Match Score:</strong> <span style="color: #ffc107;">{mentor.get('match_score', 0.0):.3f}</span></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No mentor recommendations available.")
    
    # Additional Insights
    if 'insights' in results:
        st.markdown("## 💡 Additional Insights")
        insights = results['insights']
        
        if insights.get('novelty_keywords'):
            st.markdown("**Key Novel Concepts:**")
            st.write(", ".join(insights['novelty_keywords']))
            
        if insights.get('patent_classification'):
            st.markdown("**Suggested Patent Classification:**")
            st.write(insights['patent_classification'])
            
        if insights.get('recommendations'):
            st.markdown("**Recommendations:**")
            for rec in insights['recommendations']:
                st.write(f"• {rec}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Patent Discovery & Mentor Recommender</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## Configuration")
    backend_url = st.sidebar.text_input(
        "Backend URL", 
        value="http://localhost:8000",
        help="URL of the backend API server"
    )
    
    # Test backend connection
    if st.sidebar.button("Test Backend Connection"):
        try:
            response = requests.get(f"{backend_url}/health", timeout=5)
            if response.status_code == 200:
                st.sidebar.success("✅ Backend connected successfully!")
            else:
                st.sidebar.error("❌ Backend not responding properly")
        except:
            st.sidebar.error("❌ Cannot connect to backend")
    
    # Main input section
    st.markdown("## 📝 Project Abstract Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"],
        horizontal=True
    )
    
    project_text = ""
    
    if input_method == "Text Input":
        project_text = st.text_area(
            "Enter your project abstract:",
            height=200,
            placeholder="Describe your project/research work here. Include technical details, methodology, and potential applications...",
            help="Provide a detailed description of your project for better analysis"
        )
    
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload project document",
            type=['txt', 'pdf', 'docx'],
            help="Upload a text file, PDF, or Word document containing your project description"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    project_text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    # For PDF processing, you'd need to add PyPDF2 or similar
                    st.warning("PDF processing not implemented. Please use text input or convert to .txt file.")
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    # For DOCX processing, you'd need to add python-docx
                    st.warning("DOCX processing not implemented. Please use text input or convert to .txt file.")
                
                if project_text:
                    st.success(f"File uploaded successfully! ({len(project_text)} characters)")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Analysis section
    if st.button("🔍 Analyze Project", type="primary", use_container_width=True):
        if not project_text.strip():
            st.warning("Please provide project text before analyzing.")
            return
        
        if len(project_text.strip()) < 50:
            st.warning("Project description is too short. Please provide more details for better analysis.")
            return
        
        # Show loading spinner
        with st.spinner("Analyzing your project... This may take a few moments."):
            results = call_backend_api(project_text, backend_url)
        
        if results:
            # Store results in session state for export
            st.session_state['analysis_results'] = results
            st.session_state['project_text'] = project_text
            
            # Display results
            display_results(results)
            
            # Export section
            st.markdown("## 📄 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export as PDF", use_container_width=True):
                    try:
                        pdf_buffer = generate_pdf_report(results, project_text)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer.getvalue(),
                            file_name=f"patent_analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
            
            with col2:
                if st.button("📊 Export as JSON", use_container_width=True):
                    json_data = {
                        "project_text": project_text,
                        "analysis_results": results,
                        "timestamp": str(st.session_state.get('analysis_timestamp', 'N/A'))
                    }
                    st.download_button(
                        label="Download JSON Data",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"patent_analysis_data.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    # Help section
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Input Method**: Choose between typing directly or uploading a file
        2. **Provide Details**: Enter comprehensive project description including:
           - Technical approach and methodology
           - Novel aspects and innovations
           - Potential applications and benefits
           - Current challenges and solutions
        
        3. **Analyze**: Click the analyze button to get:
           - Patentability score (0.0 to 1.0)
           - Similar existing patents
           - Recommended mentors for your domain
        
        4. **Interpret Results**:
           - **High Score (0.6+)**: Strong patent potential, proceed with filing
           - **Medium Score (0.3-0.6)**: Requires review and improvements
           - **Low Score (<0.3)**: Limited patent potential, focus on research
        
        5. **Export**: Download PDF report or JSON data for records
        
        ### Tips for Better Results:
        - Be specific about technical innovations
        - Include methodology and implementation details
        - Mention unique advantages over existing solutions
        - Provide context about the problem being solved
        """)

if __name__ == "__main__":
    main()