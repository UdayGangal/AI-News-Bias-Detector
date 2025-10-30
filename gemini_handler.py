"""
Gemini Handler - Handles all Gemini API interactions
Verifies news authenticity and generates summaries
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

class GeminiHandler:
    def __init__(self):
        """Initialize Gemini with API key from .env file"""
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def verify_news(self, news_text):
        """
        Verify if the news is true or potentially fake
        Returns: dict with verification status and analysis
        """
        prompt = f"""
        Analyze the following news article or headline and determine if it appears to be:
        1. TRUE - Based on verifiable facts and credible sources
        2. LIKELY TRUE - Plausible but needs verification
        3. QUESTIONABLE - Contains dubious claims or lacks credibility
        4. FALSE - Clearly false or misinformation
        
        News: {news_text}
        
        Provide your analysis in the following format:
        STATUS: [TRUE/LIKELY TRUE/QUESTIONABLE/FALSE]
        ANALYSIS: [Brief explanation of why you rated it this way]
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Parse the response
            lines = result_text.strip().split('\n')
            status = "UNKNOWN"
            analysis = "Unable to analyze"
            
            for line in lines:
                if line.startswith('STATUS:'):
                    status = line.replace('STATUS:', '').strip()
                elif line.startswith('ANALYSIS:'):
                    analysis = line.replace('ANALYSIS:', '').strip()
            
            return {
                'is_true': status,
                'analysis': analysis,
                'error': None
            }
        
        except Exception as e:
            return {
                'is_true': 'ERROR',
                'analysis': '',
                'error': str(e)
            }
    
    def summarize_news(self, news_text):
        """
        Generate a concise summary of the news article
        Returns: summary string
        """
        prompt = f"""
        Provide a concise, objective summary of the following news article or headline.
        Keep it brief (2-3 sentences) and focus on the key facts.
        
        News: {news_text}
        
        Summary:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"