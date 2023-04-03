import os
import pandas as pd
import openai
from openai.error import RateLimitError
from tqdm import tqdm
import time 
tqdm.pandas()



openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

def get_sentiment(text):
    while True:
        try:
            completion = openai.ChatCompletion.create(
              model = 'gpt-4',
              messages = [
                {'role': 'user', 'content': f"""You are a classifier. I will pass in a message in ukrainian or russian regarding digital education. You predict 4 classes: 
                0 -> digital education does not exist (this is in the messages often marked with -) 
                1 -> positive sentiment towards digital education
                2 -> neutral sentiment towards digital education
                3-> negative sentiment towards digital education. 
                I will now past the message you just return the class as integer.
                {text}"""},
              ],
            #   max_tokens = 32000,
              n = 1,
              stop = None,
              temperature=0.5,
              timeout=1000
            )
            return completion['choices'][0]['message']['content']
        except RateLimitError:
            print("Rate limit error, waiting 5 secs and trying again")
            time.sleep(5)
            continue

if __name__ == "__main__":
    df  = pd.read_json("messages.json")
    df['sentiment'] = df['message'].progress_apply(get_sentiment)
    df.to_csv("sentiment.csv", index=False)