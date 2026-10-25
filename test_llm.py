import sys

from llm_client import LLMRequestError, ask_llm

# Windows 控制台默认 GBK，无法输出 emoji 等字符
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    try:
        answer = ask_llm("当前环境怎么样？")
        print("大模型回复：")
        print(answer)
    except LLMRequestError as exc:
        print(f"API调用失败：{exc}")
