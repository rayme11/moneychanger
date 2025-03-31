from typing import Tuple, Dict
import dotenv
import os
from dotenv import load_dotenv
import requests
import json
import streamlit as st
from openai import OpenAI


token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)



load_dotenv()
EXCHANGERATE_API_KEY= os.getenv('EXCHANGERATE_API_KEY')




def get_exchange_rate(base: str, target: str, amount: str) -> Tuple:
    """Return a tuple of (base, target, amount, conversion_result (2 decimal places))"""
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base}/{target}/{amount}"
    response = json.loads(requests.get(url).text)
    return (base, target, amount, f'{response["conversion_result"]:.2f}')

print(get_exchange_rate("USD", "GBP", "350"))

def call_llm(textbox_input) -> Dict:
    """Make a call to the LLM with the textbox_input as the prompt.
       The output from the LLM should be a JSON (dict) with the base, amount and target"""

    tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "exchange_rate_function",
                        "description": "Convert a given amount of money from one currency to another. Each currency will be represented as a 3-letter code",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "base": {
                                    "type": "string",
                                    "description": "The base or original currency.",
                                },
                                "target": {
                                    "type": "string",
                                    "description": "The target or converted currency",
                                },
                                "amount": {
                                    "type": "string",
                                    "description": "The amount of money to convert from the base currency.",
                                },
                            },
                            "required": ["base", "target", "amount"],
                            "additionalProperties": False,
                        },
                    },
                }
                ]
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": textbox_input,
                }
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name,
            tools=tools,
        )
        
    except Exception as e:
        print(f"Exception {e} for {textbox_input}")
       
    else:
        return response#.choices[0].message.content

def run_pipeline(user_input: str) -> None:
    """Based on textbox_input, determine if you need to use the tools (function calling) for the LLM.
    Call get_exchange_rate(...) if necessary"""

    response = call_llm(user_input)
    st.write(response)

    if response.choices[0].finish_reason=='tool_calls':
        response_arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        base = response_arguments["base"]
        target = response_arguments["target"]
        amount = response_arguments["amount"]
        _, _, _, conversion_result = get_exchange_rate(base, target, amount)
        st.write(f'{base} {amount} is {target} {conversion_result}')

    elif response.choices[0].finish_reason == 'stop':
        # This is the case where the LLM has provided a response without using the function calling 
        st.write(f"(Function calling not used) and {response.choices[0].message.content}")
    else:
        st.write("NotImplemented")


# Title of the app
st.title("Multilingual Money Changer")

# Textbox for user input
user_input = st.text_input("Enter the amount and the currency")

# Submit button
if st.button("Submit"):
    #Display the input text below the textbox
    run_pipeline(user_input)
    #st.write(response)
   