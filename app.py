import base64
import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

from llm.V4 import Chatbot, AppBaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = Chatbot(api_key=os.getenv('OPENAI_API_KEY'), api_url=os.getenv('OPENAI_API_BASE'))


class StepResultModel(AppBaseModel):
    title: str
    content: str
    next_action: str
    confidence: float


def make_api_call(messages, max_tokens, temperature=0.5, is_final_answer=False, model="gpt-4o"):
    for attempt in range(3):
        try:
            logger.info(f"尝试进行API调用 (第 {attempt + 1}/3 次尝试)")
            content, _, _, _ = client.ask(
                model=model,
                prompt=messages,
                json_format=True,
                max_tokens=max_tokens,
                temperature=temperature,
                response_model=StepResultModel
            )
            logger.info("API调用成功")
            logger.info(content)
            return json.loads(content)
        except Exception as e:
            logger.error(f"API调用失败 (第 {attempt + 1}/3 次尝试)。错误: {str(e)}")
            if attempt == 2:
                if is_final_answer:
                    logger.error("3次尝试后未能生成最终答案")
                    return {
                        "title": "错误",
                        "content": f"3次尝试后未能生成最终答案。错误: {str(e)}",
                    }
                else:
                    logger.error("3次尝试后未能生成步骤")
                    return {
                        "title": "错误",
                        "content": f"3次尝试后未能生成步骤。错误: {str(e)}",
                        "next_action": "final_answer",
                    }
            logger.info("等待1秒后重试")
            time.sleep(1)  # 重试前等待1秒


def generate_response(prompt, max_steps=5, temperature=0.5, model="gpt-4o"):
    logger.info(f"正在为提示生成回答: {prompt}")
    messages = [
        {
            "role": "system",
            "content": """You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning. Follow these instructions:

1. Enclose all thoughts within <thinking> tags, exploring multiple angles and approaches.
2. Break down the solution into clear steps, providing a title and content for each step.
3. After each step, decide if you need another step or if you're ready to give the final answer.
4. Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
5. Regularly evaluate your progress, being critical and honest about your reasoning process.
6. Assign a quality score between 0.0 and 1.0 to guide your approach:
   - 0.8+: Continue current approach
   - 0.5-0.7: Consider minor adjustments
   - Below 0.5: Seriously consider backtracking and trying a different approach
7. If unsure or if your score is low, backtrack and try a different approach, explaining your decision.
8. For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
9. Explore multiple solutions individually if possible, comparing approaches in your reflections.
10. Use your thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
11. Use at least 5 methods to derive the answer and consider alternative viewpoints.
12. Be aware of your limitations as an AI and what you can and cannot do.

After every 3 steps, perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints.

Respond in JSON format with 'title', 'content', 'next_action' (either 'continue', 'reflect', or 'final_answer'), and 'confidence' (a number between 0 and 1) keys.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue",
    "confidence": 0.8
}```

Your goal is to demonstrate a thorough, adaptive, and self-reflective problem-solving process, emphasizing dynamic thinking and learning from your own reasoning.""",
        },
        {"role": "user", "content": prompt}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        logger.info(f"开始第 {step_count} 步")
        start_time = time.time()
        step_data = make_api_call(messages, 4096, temperature=temperature, model=model)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        logger.info(f"第 {step_count} 步完成。思考时间: {thinking_time:.2f} 秒")
        steps.append((f"{step_data['title']}", step_data["content"], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data["next_action"] == "final_answer" and step_count < max_steps:
            messages.append({"role": "user",
                             "content": "Please continue your analysis with at least 5 more steps before providing the final answer."})
        elif step_data["next_action"] == "final_answer":
            logger.info("已达到最终答案或最大步骤数")
            break
        elif step_data["next_action"] == 'reflect' or step_count % 3 == 0:
            messages.append({"role": "user",
                             "content": "Please perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints."})
        else:
            messages.append({"role": "user", "content": "Please continue with the next step in your analysis."})
        step_count += 1

        yield steps, None, None  # 我们现在yield三个值,但只有steps是有意义的

    # 生成最终答案
    messages.append({"role": "user",
                     "content": "Please provide a comprehensive final answer based on your reasoning above, summarizing key points and addressing any uncertainties. USE JSON Formate"})

    start_time = time.time()
    final_data = make_api_call(messages, 4096, temperature=temperature, is_final_answer=True, model=model)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if 'content' in final_data:
        final_content = final_data["content"]
    elif 'final_answer' in final_data:
        final_content = final_data["final_answer"]
    else:
        final_content = json.dumps(final_data)
    logger.info(f"最终答案已生成。思考时间: {thinking_time:.2f} 秒")
    steps.append(("最终答案", final_content, thinking_time))

    logger.info(f"总思考时间: {total_thinking_time:.2f} 秒")
    full_response = {"steps": steps, "total_thinking_time": total_thinking_time}
    yield steps, total_thinking_time, full_response


def get_binary_file_downloader_html(bin_file, file_label="文件"):
    with open(bin_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f"""
    <a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}"
       style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #4CAF50; text-decoration: none; border-radius: 4px;">
        📥 下载 {file_label}
    </a>
    """
    return href


def main():
    st.set_page_config(page_title="g1 原型", page_icon="🧠", layout="wide")

    st.title("g1: 使用 LLM 创建类似 o1 的推理链")

    st.markdown(
        """
    <style>
        /* 新样式 */
        h1, h2, h3 {
            color: #1e3a8a;
        }

        /* 侧边栏样式调整 */
        .css-1d391kg {
            padding-top: 1rem;
            padding-right: 0.5rem;
            padding-left: 0.5rem;
        }
        .css-1d391kg .block-container {
            padding-top: 1rem;
        }
        /* 调整侧边栏宽度 */
        .css-1q1n0ol {
            max-width: 14rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    这是一个早期原型,使用提示来创建类似 o1 的推理链以提高输出准确性。它并不完美,准确性尚未经过正式评估。

    开源[代码库在此](https://github.com/Theigrams/g1)
    """
    )

    with st.sidebar:
        st.markdown("## 🛠️ 设置")

        st.markdown("<br>", unsafe_allow_html=True)  # 添加间距

        st.markdown("### 🤖 模型设置")
        model_options = [
            "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307",
            "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b-exp-0924",
            "qwen2-72b-instruct", "qwen2.5-72b-instruct",
            "llama-3.1-70b-versatile"
        ]
        selected_model = st.selectbox("选择模型", model_options)
        model = selected_model

        st.markdown("<br>", unsafe_allow_html=True)  # 添加间距

        st.markdown("### ⚙️ 生成设置")
        max_steps = st.slider("最大步骤数", 3, 32, 10)
        temperature = st.slider("温度", 0.0, 1.0, 0.2, 0.1)

    # 用户查询的文本输入和发送按钮
    st.markdown("### 🔍 输入您的查询")
    col1, col2 = st.columns([5, 1])  # 创建两列，比例为 5:1
    with col1:
        user_query = st.text_input("", placeholder="例如：1.11 和 1.3 哪个大?",
                                   label_visibility="collapsed")
    with col2:
        send_button = st.button("发送")

    if send_button and user_query:
        with st.spinner("正在生成回答..."):  # 添加加载指示器
            # 创建空元素以保存生成的文本和总时间
            response_container = st.empty()
            time_container = st.empty()
            download_container = st.empty()

            # 生成并显示回答
            for steps, total_thinking_time, full_response in generate_response(
                    user_query, max_steps=max_steps, temperature=temperature, model=model
            ):
                with response_container.container():
                    for i, (title, content, thinking_time) in enumerate(steps):
                        if title.startswith("最终答案"):
                            st.markdown(f"### 🎯 {title}")
                            st.info(content)
                        else:
                            with st.expander(f"🧠 {title} (思考时间: {thinking_time:.2f} 秒)", expanded=True):
                                st.write(content)  # 使用 write 而不是 markdown 以避免 HTML 转义问题

                # 仅在结束时显示总时间
                if total_thinking_time is not None and full_response is not None:
                    time_container.markdown(f"⏱️ **总思考时间: {total_thinking_time:.2f} 秒**")

                    # 创建 JSON 文件并提供下载链接
                    json_filename = "reasoning_chain.json"
                    with open(json_filename, "w") as f:
                        json.dump(full_response, f, indent=2)

                    download_link = get_binary_file_downloader_html(json_filename, "完整推理链 JSON")
                    download_container.markdown(download_link, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
