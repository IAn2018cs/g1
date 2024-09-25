import base64
import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE'))


def extract_first_json(response):
    json_start = response.index("{")
    json_end = response.find("}")
    return json.loads(response[json_start:json_end + 1])


def extract_json(response):
    response = response.replace("JSON\n", "").replace("json\n", "").replace("```", "")
    json_start = response.index("{")
    json_end = response.rfind("}")
    return json.loads(response[json_start:json_end + 1])


def make_api_call(messages, max_tokens, temperature=0.5, is_final_answer=False, model="gpt-4o"):
    for attempt in range(3):
        try:
            logger.info(f"尝试进行API调用 (第 {attempt + 1}/3 次尝试)")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            logger.info("API调用成功")
            content = response.choices[0].message.content
            logger.info(content)
            return extract_first_json(content)
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
            "content": """你是一位专家级AI助手，能够逐步解释你的推理过程。
对于每一步，请提供一个描述该步骤内容的标题，以及具体内容。
决定是否需要另一个步骤或是否准备好给出最终答案。以 JSON 格式回应，包含 'title'、'content' 和 'next_action' (next_action 的值只有 'continue' 或 'final_answer') 。
尽可能使用多个推理步骤，至少 3 个。
请注意你作为语言模型的局限性，以及你能做和不能做的事情。在你的推理中，包括对替代答案的探索。考虑到你可能是错的，如果你的推理中有错误，错误可能在哪里。
充分测试所有其他可能性。你可能会犯错。当你说你在重新审视时，请真正地重新审视，并使用另一种方法来做到这一点。不要只是说你在重新审视。
使用至少 3 种方法来得出答案。使用最佳实践。

有效JSON响应的示例:
```json
{
    "title": "识别关键信息",
    "content": "为了开始解决这个问题，我们需要仔细检查给定的信息，并识别出将指导我们解决过程的关键元素。这涉及到...",
    "next_action": "continue"
}```

注意，每次只输出一个推理步骤，不要全部输出，请等待下一步的指令和上下文信息，再继续输出推理步骤。
""",
        },
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "谢谢！我现在将按照我的指示，从分解问题开始，逐步思考。",
        },
        {"role": "user", "content": "continue"}
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
        steps.append((f"步骤 {step_count}: {step_data['title']}", step_data["content"], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data["next_action"] == "final_answer" or step_count >= max_steps:
            logger.info("已达到最终答案或最大步骤数")
            break
        messages.append({"role": "user", "content": step_data['next_action']})
        step_count += 1

        yield steps, None, None  # 我们现在yield三个值,但只有steps是有意义的

    # 生成最终答案
    messages.append({"role": "user", "content": "请根据你上面的推理给出最终答案。注意还要以 JSON 的格式输出"})

    start_time = time.time()
    final_data = make_api_call(messages, 4096, temperature=temperature, is_final_answer=True, model=model)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    logger.info(f"最终答案已生成。思考时间: {thinking_time:.2f} 秒")
    steps.append(("最终答案", final_data["content"], thinking_time))

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

    st.title("g1: 使用 GPT-4o 创建类似 o1 的推理链")

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
            "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b-exp-0924",
            "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307",
            "qwen2-72b-instruct", "qwen2.5-72b-instruct",
            "llama-3.1-70b-versatile"
        ]
        selected_model = st.selectbox("选择模型", model_options)
        model = selected_model

        st.markdown("<br>", unsafe_allow_html=True)  # 添加间距

        st.markdown("### ⚙️ 生成设置")
        max_steps = st.slider("最大步骤数", 3, 32, 10)
        temperature = st.slider("温度", 0.0, 1.0, 0.5, 0.1)

    # 用户查询的文本输入和发送按钮
    st.markdown("### 🔍 输入您的查询")
    col1, col2 = st.columns([5, 1])  # 创建两列，比例为 5:1
    with col1:
        user_query = st.text_input("", placeholder="例如：单词'strawberry'中有多少个'R'?",
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
