import os
import pandas as pd
import openai
from openai.error import RateLimitError
from tqdm import tqdm
import time 
import tiktoken
tqdm.pandas()


openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

model_name = "gpt-4"
# encoding = tiktoken.encoding_for_model(model_name=model_name)
encoding = "cl100k_base"
max_input_tokens = 7500


# calculate number of tokens in a text string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# sample from df
def sample_df_gpt_analysis(df, max_input_tokens, sentiment):
    start_prompt = """You are a summarizer. I will pass in a messages in ukrainian or russian regarding {sentiment} sentiment towards digital education, the number at the start is the index of the message. 
    You return a summary regarding the {sentiment} sentiment of the messages. 
    Apart from the general summary, create a list of the top 10 mentioned aspects regarding the {sentiment} sentiment and refer to the indexes of the messages mentioning these aspects.
    All analysis in english please"""
    current_input_tokens = num_tokens_from_string(start_prompt, encoding_name=encoding)
    text_list = []
    text_list.append(start_prompt)
    while max_input_tokens > current_input_tokens:
        # print(len(df))
        df_sample = df.sample(n=1, replace=False)
        df = df.drop(df_sample.index)
        current_input_tokens += num_tokens_from_string(str(df_sample["message"].values), encoding_name=encoding)
        if current_input_tokens > max_input_tokens:
            break
        text = str(df_sample.index.values) + " " + df_sample["message"].values[0]
        text_list.append(text)
        if len(df) == 0:
            break
    print("number of messages analysed: ", len(text_list))
    text = '\n'.join(text_list)
    print("number of messages not analysed: ",  len(df))
    return text

def get_summarise(prompt):
    while True:
        try:
            completion = openai.ChatCompletion.create(
              model = model_name,
              messages = [
                {'role': 'user', 'content': prompt},
              ],
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
    df = pd.read_csv("sentiment.csv")
    sentiment_dict ={
        0: "no comment",
        1: "positive",
        2: "neutral",
        3: "negative"
    }
    for i in [1,2,3]:
        print("analysing sentiment: ", i)
        df_sentiment = df[df["sentiment"] == str(i)]
        sentiment = sentiment_dict[i]
        prompt = sample_df_gpt_analysis(df_sentiment, max_input_tokens, sentiment=sentiment)
        output_gpt = get_summarise(prompt)
        with open(f"output_gpt_{i}.txt", "w") as f:
             f.write(output_gpt)