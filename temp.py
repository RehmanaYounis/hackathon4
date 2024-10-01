from langchain_text_splitters import RecursiveCharacterTextSplitter 

# Set the URL and extract the video ID
url = "https://www.youtube.com/watch?v=HFfXvfFe9F8"
video_id = url.split("=")[1]

file_path = f'{video_id}.txt'
with open(file_path, 'r', encoding='UTF-8') as file:
    text_content = file.read()

length_function = len
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=length_function,
)

splits = splitter.split_text(text_content)

