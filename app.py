import json
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from litellm import completion


def make_api_call(messages, max_tokens, temperature=0.2, is_final_answer=False, model="gpt-4o"):
    for attempt in range(3):
        try:
            response = completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}",
                    }
                else:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                        "next_action": "final_answer",
                    }
            time.sleep(1)  # Wait for 1 second before retrying


def generate_response(prompt, max_steps=5, temperature=0.2, model="gpt-4o"):
    messages = [
        {
            "role": "system",
            "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue"
}```
""",
        },
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem.",
        },
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 4096, temperature=temperature, model=model)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data["content"], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data["next_action"] == "final_answer" or step_count >= max_steps:
            break

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, 4096, temperature=temperature, is_final_answer=True, model=model)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time


def main():
    st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")

    st.title("g1: Using GPT-4o to create o1-like reasoning chains")

    st.markdown(
        """
    This is an early prototype of using prompting to create o1-like reasoning chains to improve output accuracy. It is not perfect and accuracy has yet to be formally evaluated.
                
    Open source [repository here](https://github.com/Theigrams/g1)
    """
    )

    with st.sidebar:
        st.subheader("Settings")
        max_steps = st.slider("Max Steps", 3, 10, 5)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

        # Modify model selection
        model_options = ["gpt-4o", "gpt-4o-mini", "custom"]
        selected_model = st.selectbox("Select Model", model_options)

        # Add custom model input
        if selected_model == "custom":
            custom_model = st.text_input("Enter custom model name")
            model = custom_model if custom_model else "gpt-4o"  # If not entered, use gpt-4o as default
        else:
            model = selected_model

        # Add API key and API base input
        api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        api_base = st.text_input("API Base URL", value=os.getenv("OPENAI_API_BASE", ""))

        # Save API settings
        if st.button("Save API Settings"):
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = api_base
            st.success("API settings saved successfully")

    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")

    if user_query:
        with st.spinner("Generating response..."):  # Add loading indicator
            # Create empty elements to hold the generated text and total time
            response_container = st.empty()
            time_container = st.empty()

            # Generate and display the response
            for steps, total_thinking_time in generate_response(
                user_query, max_steps=max_steps, temperature=temperature, model=model
            ):
                with response_container.container():
                    for i, (title, content, thinking_time) in enumerate(steps):
                        if title.startswith("Final Answer"):
                            st.subheader(title)  # Use subheader instead of markdown
                            st.write(content)
                        else:
                            with st.expander(f"{title} (Thinking time: {thinking_time:.2f} seconds)", expanded=True):
                                st.write(content)  # Use write instead of markdown to avoid HTML escaping issues

                # Only show total time when it's available at the end
                if total_thinking_time is not None:
                    time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")


if __name__ == "__main__":
    main()
