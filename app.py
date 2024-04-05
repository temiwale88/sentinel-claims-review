import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
from openai import OpenAI

file_path = 'prompts.npy'

st.cache_resource
def load_and_preprocess_prompt_data():
    file_path = 'prompts.npy'

    # Load the DataFrame back from the binary file
    prompt_df = np.load(file_path, allow_pickle=True)
    prompt_df = pd.DataFrame(prompt_df, columns=['member_id', 'prompt'])
    
    return prompt_df

# Function to load and preprocess data (cached)
st.cache_resource
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("data.csv", encoding='ISO 8859-1')
    df['Service Date'] = pd.to_datetime(df['Service Date'])

    # Sort DataFrame by 'Member ID' and 'Service Date' in descending order
    df.sort_values(by=['Member ID', 'Service Date'], ascending=[True, False], inplace=True)
    df.columns = df.columns.str.lower()
    df['member id'] = df['member id'].astype(str)
    df['policy'] = df['procedure code'] + ": " + df['procedure guidelines']
    df.drop(['procedure guidelines'], axis=1, inplace=True)
    
    return df

from tabulate import tabulate # first, we'll convert to markdown

# Source code (modified slightly) from 'laura-xy-lee' from: https://tinyurl.com/3fr42jky
def convert_to_md(df):
        # Convert table to markdown
        md = tabulate(df, tablefmt='github', headers='keys', showindex=False)
        return md


# Function to create prompt for OpenAI ChatGPT
def create_prompt(prompt):
    return [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]

# Function to generate reply from OpenAI ChatGPT
def generate_reply(prompt):
    try:
        client = OpenAI()

        stream = client.chat.completions.create(
            model="gpt-4",
            messages=create_prompt(prompt),
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=True,
        )
        reply = ""
        for chunk in stream:
            reply += chunk.choices[0].delta.content or ""
        return reply
    except Exception as e:
        return "Sorry. There was an issue generating a reply."

# Streamlit app
def main():
    st.title('Explore Fraud, Waste, and Abuse Investigation with Sentinel')
    st.sidebar.write(f"""
    ### Description
    This application is a Generative AI proof-of-concept. It features ***Sentinel***, a tool designed to accelerate the claims and utilization management review process. Sentinel leverages Generative AI models to analyze data, detect anomalies, and provide recommendations to team members for efficient decision-making. It is in essence a **co-pilot**. **Note** that this is ***not production-grade*** but a simple deployment for a demonstration.
    
    ---
    
    ### App info

    **App name**: *Sentinel - Your Claims Review Co-Pilot*
    
    **How to Use**: Select a member ID via the `Select Member ID:` option in the main view, then click on `Explore Sentinel's Analysis`. Explore the claims history for this fictitious member and wait for Sentinel's analysis and recommendations.🎉
    
    ---

    **How do I make something like this for my organization?**

    Contact **Elijah Adeoye** (co-founder @ *Unveil*) via email [Elijah Adeoye](mailto:elijah.adeoye@weunveil.ai)
    or [LinkedIn](https://www.linkedin.com/in/elijahaadeoye/)!
    """)
    # Load and preprocess prompt data
    prompt_df = load_and_preprocess_prompt_data()
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Get user input
    selected_member_id = st.selectbox('Select Member ID:', prompt_df['member_id'])

    # Generate response when button is clicked
    if st.button("Explore Sentinel's Analysis"):
        st.write("Sentinel is thinking...")
    # Filter DataFrame for selected member ID
        filtered_df = df[df['member id'] == selected_member_id][[col for col in df.columns if col !='policy']]
        filtered_df['line_number'] = filtered_df.index + 1
        st.table(filtered_df)
        # Filter DataFrame for selected member ID
        filtered_df = prompt_df[prompt_df['member_id'] == selected_member_id]
        
        # Ensure there is at least one prompt for the selected member ID
        if not filtered_df.empty:
            # Extract the prompt
            prompt = prompt_df[prompt_df['member_id'] == selected_member_id]['prompt'].values[0]
            # print(prompt)
            # Generate reply
            reply = generate_reply(prompt)
            
            # Display reply
            st.markdown("### Sentinel's Analysis:")
            st.markdown(reply)
        else:
            st.write("No prompts available for the selected member ID.")
            

if __name__ == '__main__':
    main()
