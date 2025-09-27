import io
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.colors import HexColor

class PatentAnalysisReport:
    """Generate comprehensive PDF reports for patent analysis results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1f4e79'),
            alignment=TA_CENTER,
            spaceAfter=30,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2c5aa0'),
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=HexColor('#2c5aa0'),
            borderPadding=5
        ))
        
        # Score style
        self.styles.add(ParagraphStyle(
            name='ScoreStyle',
            parent=self.styles['Normal'],
            fontSize=18,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=10
        ))
        
        # Card style for patents and mentors
        self.styles.add(ParagraphStyle(
            name='CardContent',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20,
            borderWidth=1,
            borderColor=HexColor('#e0e0e0'),
            borderPadding=10
        ))
    
    def _get_score_color(self, score: float) -> colors.Color:
        """Get color based on patentability score"""
        if score >= 0.7:
            return HexColor('#dc3545')  # Red
        elif score >= 0.4:
            return HexColor('#fd7e14')  # Orange
        else:
            return HexColor('#28a745')  # Green
    
    def _get_score_interpretation(self, score: float) -> str:
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
    
    def _create_score_chart(self, score: float) -> Drawing:
        """Create a visual representation of the score"""
        drawing = Drawing(400, 100)
        
        # Background bar
        bg_rect = Rect(50, 30, 300, 40)
        bg_rect.fillColor = HexColor('#f0f0f0')
        bg_rect.strokeColor = HexColor('#cccccc')
        drawing.add(bg_rect)
        
        # Score bar
        score_width = 300 * min(score, 1.0)
        score_rect = Rect(50, 30, score_width, 40)
        score_rect.fillColor = self._get_score_color(score)
        score_rect.strokeColor = None
        drawing.add(score_rect)
        
        return drawing
    
    def _format_project_text(self, text: str, max_length: int = 2000) -> str:
        """Format and truncate project text for display"""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        last_sentence = truncated.rfind('.')
        if last_sentence > max_length * 0.7:  # If we can find a sentence end
            return truncated[:last_sentence + 1] + "\n\n[Text truncated for brevity...]"
        else:
            return truncated + "...\n\n[Text truncated for brevity...]"
    
    def generate_report(self, results: Dict[Any, Any], project_text: str) -> io.BytesIO:
        """Generate complete PDF report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=1*inch,
            bottomMargin=1*inch,
            leftMargin=0.8*inch,
            rightMargin=0.8*inch
        )
        
        story = []
        
        # Header
        story.append(Paragraph("Patent Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Metadata table
        metadata_data = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Analysis Type:', 'Patentability Assessment & Mentor Recommendation'],
            ['Project Length:', f"{len(project_text)} characters"]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        score = results.get('patentability_score', 0.0)
        score_text = self._get_score_interpretation(score)
        
        summary_text = f"""
        The patent analysis has been completed with a patentability score of {score:.3f} out of 1.000, 
        which indicates: <b>{score_text}</b>. The system identified {len(results.get('similar_patents', []))} 
        similar patents in the database and recommended {len(results.get('recommended_mentors', []))} 
        potential mentors for this project domain.
        """
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Patentability Score Section
        story.append(Paragraph("Patentability Analysis", self.styles['SectionHeader']))
        
        # Score display
        score_color = self._get_score_color(score)
        score_paragraph = Paragraph(
            f'<font color="{score_color}">Patentability Score: {score:.3f}</font>',
            self.styles['ScoreStyle']
        )
        story.append(score_paragraph)
        
        # Score chart
        score_chart = self._create_score_chart(score)
        story.append(score_chart)
        story.append(Spacer(1, 10))
        
        story.append(Paragraph(f"<b>Interpretation:</b> {score_text}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Score breakdown if available
        if 'score_components' in results:
            story.append(Paragraph("Score Components", self.styles['Heading3']))
            
            components_data = [['Component', 'Score', 'Weight']]
            for component, value in results['score_components'].items():
                components_data.append([
                    component.replace('_', ' ').title(),
                    f"{value:.3f}",
                    "N/A"  # Add weight if available in your system
                ])
            
            components_table = Table(components_data, colWidths=[3*inch, 1*inch, 1*inch])
            components_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f9fa'), colors.white]),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ]))
            
            story.append(components_table)
            story.append(Spacer(1, 20))
        
        # Similar Patents Section
        story.append(Paragraph("Similar Patents Found", self.styles['SectionHeader']))
        
        similar_patents = results.get('similar_patents', [])
        if similar_patents:
            for i, patent in enumerate(similar_patents[:5], 1):
                patent_title = f"#{i}. {patent.get('title', 'Patent Title Not Available')}"
                story.append(Paragraph(patent_title, self.styles['Heading4']))
                
                patent_info = f"""
                <b>Patent ID:</b> {patent.get('patent_id', 'N/A')}<br/>
                <b>Similarity Score:</b> {patent.get('similarity', 0.0):.3f}<br/>
                <b>Abstract:</b> {patent.get('abstract', 'Abstract not available')[:300]}...
                """
                
                story.append(Paragraph(patent_info, self.styles['CardContent']))
                story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No similar patents found in the database.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommended Mentors Section
        story.append(Paragraph("Recommended Mentors", self.styles['SectionHeader']))
        
        mentors = results.get('recommended_mentors', [])
        if mentors:
            for i, mentor in enumerate(mentors[:3], 1):
                mentor_title = f"Mentor #{i}: {mentor.get('name', 'Name Not Available')}"
                story.append(Paragraph(mentor_title, self.styles['Heading4']))
                
                mentor_info = f"""
                <b>Domain:</b> {mentor.get('domain', 'Not specified')}<br/>
                <b>Experience:</b> {mentor.get('experience_years', 'N/A')} years<br/>
                <b>Contact:</b> {mentor.get('email', 'Not available')}<br/>
                <b>Match Score:</b> {mentor.get('match_score', 0.0):.3f}<br/>
                <b>Bio:</b> {mentor.get('bio', 'Biography not available')[:200]}...
                """
                
                story.append(Paragraph(mentor_info, self.styles['CardContent']))
                story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No mentor recommendations available.", self.styles['Normal']))
        
        # Additional Insights
        if 'insights' in results:
            story.append(PageBreak())
            story.append(Paragraph("Additional Insights", self.styles['SectionHeader']))
            
            insights = results['insights']
            
            if insights.get('novelty_keywords'):
                story.append(Paragraph("Key Novel Concepts", self.styles['Heading3']))
                keywords_text = ", ".join(insights['novelty_keywords'])
                story.append(Paragraph(keywords_text, self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            if insights.get('patent_classification'):
                story.append(Paragraph("Suggested Patent Classification", self.styles['Heading3']))
                story.append(Paragraph(insights['patent_classification'], self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            if insights.get('recommendations'):
                story.append(Paragraph("Recommendations", self.styles['Heading3']))
                for rec in insights['recommendations']:
                    story.append(Paragraph(f"• {rec}", self.styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Project Description (Appendix)
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Project Description", self.styles['SectionHeader']))
        
        formatted_text = self._format_project_text(project_text)
        story.append(Paragraph(formatted_text, self.styles['Normal']))
        
        # Footer information
        story.append(Spacer(1, 30))
        footer_text = """
        <i>This report was generated by the Patent Discovery & Mentor Recommender system. 
        The patentability scores and recommendations are based on AI analysis and should be 
        reviewed by patent experts before making final decisions.</i>
        """
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer

def generate_pdf_report(results: Dict[Any, Any], project_text: str) -> io.BytesIO:
    """
    Main function to generate PDF report
    
    Args:
        results: Dictionary containing analysis results from backend
        project_text: Original project description text
        
    Returns:
        io.BytesIO: PDF report as bytes buffer
    """
    
    try:
        report_generator = PatentAnalysisReport()
        return report_generator.generate_report(results, project_text)
    
    except Exception as e:
        # Create a simple error report if main generation fails
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        story = [
            Paragraph("Patent Analysis Report - Error", styles['Title']),
            Spacer(1, 20),
            Paragraph(f"An error occurred while generating the report: {str(e)}", styles['Normal']),
            Spacer(1, 20),
            Paragraph("Raw Results:", styles['Heading2']),
            Paragraph(str(results), styles['Normal'])
        ]
        
        doc.build(story)
        buffer.seek(0)
        return buffer

# Utility functions for different export formats
def generate_json_export(results: Dict[Any, Any], project_text: str) -> Dict[str, Any]:
    """Generate JSON export of analysis results"""
    return {
        "timestamp": datetime.now().isoformat(),
        "project_description": project_text,
        "analysis_results": results,
        "export_metadata": {
            "format_version": "1.0",
            "system": "Patent Discovery & Mentor Recommender"
        }
    }

def generate_csv_summary(results: Dict[Any, Any]) -> str:
    """Generate CSV summary of key metrics"""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow(["Metric", "Value", "Interpretation"])
    
    # Patentability score
    score = results.get('patentability_score', 0.0)
    if score >= 0.8:
        interpretation = "Very High - Strong patent potential"
    elif score >= 0.6:
        interpretation = "High - Good patent potential"
    elif score >= 0.4:
        interpretation = "Medium - Requires review"
    elif score >= 0.2:
        interpretation = "Low - Limited patent potential"
    else:
        interpretation = "Very Low - Unlikely to be patentable"
    
    writer.writerow(["Patentability Score", f"{score:.3f}", interpretation])
    
    # Similar patents count
    similar_count = len(results.get('similar_patents', []))
    writer.writerow(["Similar Patents Found", similar_count, f"{similar_count} patents in database"])
    
    # Mentor recommendations count
    mentor_count = len(results.get('recommended_mentors', []))
    writer.writerow(["Mentors Recommended", mentor_count, f"{mentor_count} suitable mentors"])
    
    # Top similarity score
    if results.get('similar_patents'):
        top_similarity = max(patent.get('similarity', 0) for patent in results['similar_patents'])
        writer.writerow(["Highest Similarity", f"{top_similarity:.3f}", "Most similar patent found"])
    
    return output.getvalue()