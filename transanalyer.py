import warnings 
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from langchain_community import output_parsers
from langchain_core.pydantic_v1 import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import os
load_dotenv()

class ReviewClassification(BaseModel):
    """Pydantic model to get the sentiment of the excerpt"""
    sentiment: str = Field(description='The sentiment of the text', enum=["positive", "negative", "neutral"])
    # language: str= Field (..., enum=["Roman Urdu", "Spanish", "English"])
    # topic: str=Field(description="Holds information on the transcript ")
            
@dataclass
class Transanalyer:
    url: str = ""
    transcript: str = ""
    chunks: List[str] = dataclasses.field(default_factory=list)
    res: Dict[str, int] = dataclasses.field(default_factory=lambda: {"positive": 0, "negative": 0, "neutral": 0})
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
            with open (f'{video_id}.txt', 'w', encoding='utf-8') as file:
                file.write(self.transcript)
                
            return self.transcript

        except Exception as e:
            raise e
    
    def get_chunks(self):   
        """
        Setting the self.chunks 
        """
        video_id=self.url.split("=")[1]
        with open(f'{video_id}.txt', 'r', encoding='UTF-8') as file:
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
            sentiment is positive, negative, or neutral in one word. MUST choose one sentiment \
            No option for None {chunk}'
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash').with_structured_output(ReviewClassification)
        chain = prompt | model 
        resp = [chain.invoke({"chunk": chunk}) for chunk in self.chunks]        
        for x, passage in zip(resp, self.chunks):
            if x and (x.sentiment.lower() != ''):
                self.res[x.sentiment.lower()] = self.res.get(x.sentiment, 0) + 1
                print(passage, x.sentiment)
        
    
    def pipeline(self, url):
        """
        Complete pipeline to do transcript analysis
        @params-input: 
            url: Youtube video URL
        @params-return:
            res: sentiment dictionary
        """
        self.get_transcription(url)
        self.get_chunks()
        self.get_sentiment()
        return self.res
    
    
if __name__ == '__main__':
    analyzer = Transanalyer()
    res = analyzer.pipeline("https://www.youtube.com/watch?v=zhWDdy_5v2ws")
    print(res)