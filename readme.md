# Speech GPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple command-line tool that allows you to chat with your speeches.

## Features

- Speech to text using OpenAI Whisper
- VectorDB for storing speeches

## Prerequisites

- Python 3.10
- FFMPEG v4+

## Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/conceptcodes/speech-gpt-python.git
   cd speech-gpt-python
   ```

2. Create a virtual environment:

   ```sh
    conda create -n speech-gpt python=3.10
    conda activate speech-gpt
   ```

3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Grab an API key from [OpenAI](https://beta.openai.com/)

5. export the API key to your environment

   ```sh
   export OPENAI_API_KEY=<your-api-key>
   ```


## Usage

To run the CLI, use the following command:

```sh
python main.py -f /path/to/file


     _______..______    _______  _______   ______  __    __  
    /       ||   _  \  |   ____||   ____| /      ||  |  |  | 
   |   (----`|  |_)  | |  |__   |  |__   |  ,----'|  |__|  | 
    \   \    |   ___/  |   __|  |   __|  |  |     |   __   | 
.----)   |   |  |      |  |____ |  |____ |  `----.|  |  |  | 
|_______/    | _|      |_______||_______| \______||__|  |__| 
                                                             
  _______ .______   .___________.
 /  _____||   _  \  |           |
|  |  __  |  |_)  | `---|  |----`
|  | |_ | |   ___/      |  |     
|  |__| | |  |          |  |     
 \______| | _|          |__|     
                                 

Model loaded!
/opt/anaconda3/envs/speech-gpt/lib/python3.10/site-packages/whisper/transcribe.py:126: UserWarning: FP16 is not supported on CPU; using FP32 instead
  warnings.warn("FP16 is not supported on CPU; using FP32 instead")
Transcription done!


>>> give me a summary

Answer: The context provided is very unclear and contains a lot of unrelated information and slang. It appears to be discussing Einstein's theory of special relativity, specifically the concept of relative motion and the idea that time can be affected by speed. However, it is difficult to provide a clear and concise summary based on the given context.
```

## Roadmap

- [ ] Store embeddings in local postgres DB
- [ ] Add a cache for files
- [ ] clean up the CLI
- [ ] Add timestamp sources to the response
