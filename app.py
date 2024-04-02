import gradio as gr
from langchain.document_loaders import UnstructuredURLLoader
import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from gtts import gTTS
from IPython.display import Audio

# from dotenv import load_dotenv 
# load_dotenv()  # to load all the env variables

api_key = os.getenv("HUGGINGFACE_API_TOKEN")

os.environ['HUGGINGFACEHUB_API_TOKEN']  = api_key


def get_summary_from_url(url):
    loaders = UnstructuredURLLoader(urls=[url])
    data = loaders.load()
    question = data[0]
    
    template = """{question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    llm_chain = LLMChain(prompt=prompt,
                         llm=HuggingFaceHub(repo_id="Mr-Vicky-01/conversational_sumarization",
                                            model_kwargs={"max_length":100,
                                                          "max_new_tokens":100,
                                                          "do_sample": False}))
    answer = llm_chain.run(question)
    return answer

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    return "output.mp3"

def summarize_and_convert_to_audio(url):
    summary = get_summary_from_url(url)
    audio_file = text_to_speech(summary)
    return audio_file

example = [
    ["https://en.wikipedia.org/wiki/Vijay_(actor)"],
    ["https://en.wikipedia.org/wiki/Sam_Altman"],
    ["https://timesofindia.indiatimes.com/india/air-india-sacks-pilot-found-drunk-after-operating-overseas-flight/articleshow/108829830.cms"]
]

iface = gr.Interface(fn=summarize_and_convert_to_audio, inputs="text", outputs="audio", title="Text Summarization & Audio Generation", description="Enter the URL of the article to summarize and convert to audio.", examples=example)
iface.launch()