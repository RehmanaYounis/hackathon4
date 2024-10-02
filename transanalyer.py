import warnings 
warnings.filterwarnings('ignore')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from typing import List, Dict
import dataclasses, os

from dotenv import load_dotenv
load_dotenv()

class ReviewClassification(BaseModel):
    """Pydantic model to get the sentiment of the excerpt"""
    sentiment: str = Field(description='The sentiment of the text', enum=["excited","good", "neutral","need intensity", "not good"])
            
@dataclass
class Transanalyer:
    url: str = ""
    transcript: str = ""
    chunks: List[str] = dataclasses.field(default_factory=list)
    res: Dict[str, int] = dataclasses.field(default_factory=lambda: {"excited":0,"good":0, "neutral":0,"need intensity":0, "not good":0})
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    def __repr__(self):
        return "success"
    
    def get_transcription(self, url):
        """
        Setting the self.transcript and self.url
        """
        self.url = url
        try:
            video_id=url.split("=")[1]
            transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]
            self.transcript = transcript
            os.makedirs('transcripts', exist_ok=True)
            with open (f'transcripts/{video_id}.txt', 'w', encoding='utf-8') as file:
                file.write(self.transcript)
                
            return self.transcript

        except Exception as error:
            raise error
    
    def get_chunks(self):   
        """
        Setting the self.chunks 
        """
        video_id=self.url.split("=")[1]
        with open(f'transcripts/{video_id}.txt', 'r', encoding='UTF-8') as file:
            text_content = file.read()

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
        )

        self.chunks = splitter.split_text(text_content)
        
    def get_sentiment(self):
        """
        Setting the self.res dictionary
        """
        prompt_template ='You are a highly accurate sentiment analysis system. Analyze the \
            sentiment of the following excerpt and provide a clear assessment of whether the \
            sentiment is "excited","good", "neutral","need intensity", "not good" in one word. MUST choose one sentiment \
            No option for None {chunk}'
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash').with_structured_output(ReviewClassification)
        chain = prompt | model 
        resp = [chain.invoke({"chunk": chunk}) for chunk in self.chunks]        
        for x in resp:
            if x and (x.sentiment.lower() != ''):
                self.res[x.sentiment.lower()] = self.res.get(x.sentiment, 0) + 1
        
    
    def pipeline(self, url):
        """
        Complete pipeline to do transcript analysis
        @params-url: str 
            Youtube video URL
        @params-return: Dict[str, int]
            sentiment dictionary
        """
        self.get_transcription(url)
        self.get_chunks()
        self.get_sentiment()
        return self.res
    
    
if __name__ == '__main__':
    analyzer = Transanalyer()
    res = analyzer.pipeline("https://www.youtube.com/watch?v=zhWDdy_5v2ws")
    print(res)