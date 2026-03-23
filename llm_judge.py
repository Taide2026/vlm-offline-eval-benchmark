import os
import re
import logging
import argparse
from dotenv import load_dotenv
from openai import OpenAI


def setup_logging(output_filename):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(output_filename, encoding="utf-8", mode="w"),
            logging.StreamHandler()
        ]
    )


def parse_log_file(log_path):
    if not os.path.exists(log_path):
        logging.error(f"找不到日誌檔案: {log_path}")
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(
        r"\[\d+/\d+\]\s*處理影片:\s*(.+?)\n"
        r".*?"
        r"模型回答\s*:\s*(.+?)\n"
        r"-{40}",
        re.DOTALL
    )

    matches = pattern.findall(content)
    parsed_items = []

    for video_path, answer in matches:
        video_path = video_path.strip()
        clean_answer = answer.strip()

        label = os.path.basename(os.path.dirname(video_path))
        if not label or label in [".", ".."]:
            label = "Unknown Action"
        else:
            label = label.replace("_", " ")

        parsed_items.append({
            "video": video_path,
            "answer": clean_answer,
            "label": label
        })

    return parsed_items


def evaluate_with_gpt5(client, answer, label):
    judge_prompt = f"""
You are an expert and impartial evaluator for Vision-Language Models (VLMs).

A VLM was asked:
"Describe the main action accurately in under 10 words."

Ground Truth Action:
{label}

VLM Output:
{answer}

Evaluate the output using these criteria:
1. Does it correctly identify the core action?
2. Is it concise and relevant?
3. Does it avoid hallucinations or unrelated details?

Return exactly in this format:
Score: [0-10]
Reason: [brief explanation in under 20 words]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Judge 評估失敗: {e}"


def main():
    parser = argparse.ArgumentParser(description="Use GPT to judge VLM results from a log file.")
    parser.add_argument("--video_dir", type=str, default="./dataset/climbing_stair")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it")
    args = parser.parse_args()

    clean_video_dir = os.path.normpath(args.video_dir)
    ground_truth_name = os.path.basename(clean_video_dir)
    if not ground_truth_name or ground_truth_name == ".":
        ground_truth_name = "default_ground_truth"

    model_name = args.model_id.split("/")[-1]

    log_file_path = os.path.join(model_name, f"{ground_truth_name}.log")
    output_log_file = os.path.join(model_name, f"{ground_truth_name}_judge_results.log")

    os.makedirs(model_name, exist_ok=True)
    setup_logging(output_log_file)

    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logging.error("❌ 找不到 OPENAI_API_KEY！")
        return

    client = OpenAI(api_key=api_key)

    logging.info(f"開始解析紀錄檔: {log_file_path}")
    items_to_judge = parse_log_file(log_file_path)
    logging.info(f"成功解析出 {len(items_to_judge)} 筆待評估紀錄。")

    if not items_to_judge:
        logging.warning("沒有找到任何可評估的內容。")
        return

    logging.info("\n🚀 開始執行 LLM-as-a-Judge 評估...\n")

    for i, item in enumerate(items_to_judge):
        video = item["video"]
        answer = item["answer"]
        label = item["label"]

        logging.info(f"[{i+1}/{len(items_to_judge)}] 正在評估影片: {video}")
        logging.info(f"Ground Truth: {label}")
        logging.info(f"🤖 VLM 回答內容: {answer}")

        judge_result = evaluate_with_gpt5(client, answer, label)

        logging.info(f"📝 Judge 評語:\n{judge_result}")
        logging.info("-" * 50)


if __name__ == "__main__":
    main()
