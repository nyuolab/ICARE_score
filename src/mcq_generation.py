# src/mcq_generation.py
import requests
from requests.exceptions import Timeout
import json
import pandas as pd
import os
from datetime import datetime
import re
import argparse
import random
import copy
import requests
import json
import os
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from config import Config
from utils import ensure_dir

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def make_llama_request(
    prompt: str,
    url: str = Config.API_URL,
    api_key: str = Config.API_KEY,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    temperature: float = Config.DEFAULT_TEMPERATURE,
    timeout: int = Config.DEFAULT_TIMEOUT,
    model: str = Config.MODEL_NAME,
    seed: int = Config.DEFAULT_SEED,
    top_p: float = Config.DEFAULT_TOP_P,
    n: int = Config.DEFAULT_N,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    frequency_penalty: float = 0.0
) -> Optional[Dict[str, Any]]:
    """Make a request to the LLAMA API."""
    if Config.API_AUTH_HEADER_TYPE == "bearer":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    elif Config.API_AUTH_HEADER_TYPE in ("api-key", "apikey"):
        header_name = "api-key" if Config.API_AUTH_HEADER_TYPE == "api-key" else "apiKey"
        headers = {
            header_name: api_key,
            "accept": "application/json",
            "Content-Type": "application/json"
        }
    else:
        raise ValueError(f"Unsupported API_AUTH_HEADER_TYPE: {Config.API_AUTH_HEADER_TYPE}")
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        # "stream": stream,
        # "stop": stop,
        "seed": seed,
        # "frequency_penalty": frequency_penalty
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in API call: {e}")
        return None
def parse_mcq(mcq_text):
    """Parse MCQ text into structured format."""
    questions = []
    
    # Split the text into individual questions
    raw_questions = mcq_text.split('\n\n')
    
    for i, raw_q in enumerate(raw_questions, start=0):
        if not raw_q.strip():
            continue
        
        try:
            # Initialize question dictionary
            question = {
                "question_id": i,
                "question_text": None,
                "options": {
                    "A": None, "B": None, "C": None, "D": None
                },
                "correct_answer": None
            }
            
            # Extract question number and text
            q_match = re.search(r'\*\*(\d+):\s+(.+?)\*\*', raw_q)
            if q_match:
                question["question_text"] = q_match.group(2).strip()
            
            # Extract options
            options = re.findall(r'([A-D])\)\s+(.+?)(?=\n[A-D]\)|$|\nAnswer:)', raw_q, re.DOTALL)
            for opt_letter, opt_text in options:
                question["options"][opt_letter] = opt_text.strip()
            
            # Extract answer
            answer_match = re.search(r'Answer:\s+([A-D])', raw_q)
            if answer_match:
                question["correct_answer"] = answer_match.group(1)
            
            # Only add complete questions
            if (question["question_text"] and 
                all(question["options"].values()) and 
                question["correct_answer"]):
                questions.append(question)
        except Exception as e:
            print(f"Error parsing question: {e}")
            continue
    
    return questions

def build_mcq_prompt(report, batch_n, previous_questions):
    """Build the MCQ-generation prompt.

    Identical to the original single-shot prompt when there are no
    previous_questions. Otherwise prepends the already-asked questions and
    adds one instruction to cover what's missing -- every other word of the
    original instructions is unchanged.
    """
    prefix = ""
    extra_instruction = ""
    if previous_questions:
        prefix = "Questions already generated for this report:\n" + \
                 "\n".join(f"- {q}" for q in previous_questions) + "\n\n"
        extra_instruction = "Do not repeat the questions listed above; cover parts of the report not addressed by them."

    return (
        prefix +
        f"Please generate {batch_n} different multiple choice question answer pairs for the following radiology report: {report}. "
        "The questions should be based on report and cannot be answered without the report."
        f"{extra_instruction}"
        "Please use the following format exactly as your life depends on sticking to these formats.:\n\n"
        "**1: [Question text]**\n"
        "A) [Option A]\n"
        "B) [Option B]\n"
        "C) [Option C]\n"
        "D) [Option D]\n"
        "Answer: [Correct answer]\n\n"
    )


def generate_mcqs_sequential(report, num_ques, batch_size, stop_threshold, url, api_key,
                              timeout, max_tokens, temperature, top_p, n, seed, model_name):
    """Generate MCQs for one report in sequential, context-aware batches.

    Each round asks for up to `batch_size` new questions and is shown the
    questions already collected so far. Stops once `num_ques` is reached, or
    once `stop_threshold` consecutive rounds are "stalled" -- either the
    model attempted fewer raw question blocks than requested (checked before
    format-validation, so a formatting slip in an otherwise content-rich
    round isn't mistaken for the report having run out of distinct content),
    or it attempted a full batch but none of it parsed into a valid
    question (guards against a report whose completions consistently fail
    to parse looping forever without ever tripping the raw-count check).
    """
    collected = []
    consecutive_short = 0

    while len(collected) < num_ques:
        batch_n = min(batch_size, num_ques - len(collected))
        prompt = build_mcq_prompt(report, batch_n, [q["question_text"] for q in collected])

        try:
            response = make_llama_request(
                prompt=prompt,
                url=url,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                model=model_name,
                seed=seed,
                top_p=top_p,
                n=n,
                stream=False,
            )
        except Timeout:
            print(f"Request timed out for report: {report[:50]}...")
            break
        except Exception as e:
            print(f"Error processing report: {e}")
            break

        if not response:
            break

        mcq_text = response["choices"][0]["message"]["content"]
        raw_blocks = [b for b in mcq_text.split('\n\n') if b.strip()]

        parsed_mcqs = parse_mcq(mcq_text)
        valid_new = [
            mcq for mcq in parsed_mcqs
            if mcq["question_text"] and all(mcq["options"].values()) and mcq["correct_answer"]
        ][:batch_n]

        for mcq in valid_new:
            mcq["question_id"] = len(collected)
            collected.append(mcq)

        # A round is "stalled" if it under-attempted (raw signal: model
        # believes there's less content left) OR made zero real progress
        # (no valid questions added, even if it attempted a full batch --
        # otherwise a report whose completions consistently fail to parse
        # could attempt a full batch every round forever without ever
        # tripping the raw-count check).
        stalled = (len(raw_blocks) < batch_n) or (len(valid_new) == 0)
        consecutive_short = consecutive_short + 1 if stalled else 0

        print(f"    round: requested={batch_n} raw_attempted={len(raw_blocks)} valid_added={len(valid_new)} "
              f"collected={len(collected)}/{num_ques} stalled={stalled} consecutive_stalled={consecutive_short}",
              flush=True)

        if consecutive_short >= stop_threshold:
            break

    return collected


def generate_and_write_mcqs(reports, num_ques, output_file, url=Config.API_URL,
    api_key=Config.API_KEY, timeout=Config.GENERATION_TIMEOUT, max_tokens=Config.GENERATION_MAX_TOKENS,
    temperature=Config.DEFAULT_TEMPERATURE, top_p=Config.DEFAULT_TOP_P, n=Config.DEFAULT_N, seed=Config.DEFAULT_SEED, model_name=Config.MODEL_NAME,
    use_sequential_generation=False, sequential_batch_size=10, sequential_stop_threshold=2):
    """Generate MCQs for given reports and write them to file."""
    try:
        # Set base random seed
        total_mcqs = 0
        total_reports_processed = 0

        with open(output_file, 'w') as file:
            # Write the opening of the JSON structure
            file.write('{\n"metadata": {\n')
            file.write(f'"generation_timestamp": "{datetime.now().isoformat()}",\n')
            file.write(f'"total_reports": {len(reports)},\n')
            file.write('"total_mcqs": 0\n},\n')
            file.write('"mcq_data": [\n')

            for i, report in enumerate(reports):
                print(f"[report {i+1}/{len(reports)}]", flush=True)
                if use_sequential_generation:
                    formatted_mcqs = generate_mcqs_sequential(
                        report, num_ques, sequential_batch_size, sequential_stop_threshold,
                        url, api_key, timeout, max_tokens, temperature, top_p, n, seed, model_name
                    )
                else:
                    # Generate MCQs until we have num_ques in the desired format
                    formatted_mcqs = []
                    while len(formatted_mcqs) < num_ques:
                        messages = [{
                            "role": "user",
                            "content": (
                                f"Please generate {num_ques} different multiple choice question answer pairs for the following radiology report: {report}. "
                                "The questions should be based on report and cannot be answered without the report."
                                "Please use the following format exactly as your life depends on sticking to these formats.:\n\n"
                                "**1: [Question text]**\n"
                                "A) [Option A]\n"
                                "B) [Option B]\n"
                                "C) [Option C]\n"
                                "D) [Option D]\n"
                                "Answer: [Correct answer]\n\n"
                            )
                        }]

                        try:
                            response = make_llama_request(
                                prompt=messages[0]["content"],
                                url=url,
                                api_key=api_key,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                timeout=timeout,
                                model=model_name,
                                seed=seed,
                                top_p=top_p,
                                n=n,
                                stream=False,
                                # stop="string",
                                # frequency_penalty=0
                            )

                            if response:
                                mcq_data = response["choices"][0]["message"]["content"]
                                parsed_mcqs = parse_mcq(mcq_data)

                                # Only add MCQs that are in the desired format
                                for mcq in parsed_mcqs:
                                    if (mcq["question_text"] and
                                        all(mcq["options"].values()) and
                                        mcq["correct_answer"]):
                                        formatted_mcqs.append(mcq)

                                    if len(formatted_mcqs) >= num_ques:
                                        break

                        except Timeout:
                            print(f"Request timed out for report: {report[:50]}...")
                            continue
                        except Exception as e:
                            print(f"Error processing report: {e}")
                            continue

                # Legacy mode requires exactly num_ques (unchanged behavior);
                # sequential mode keeps whatever was collected, even if partial.
                should_write = (len(formatted_mcqs) > 0) if use_sequential_generation else (len(formatted_mcqs) == num_ques)
                if should_write:
                    report_data = {
                        "report": report,
                        "questions": formatted_mcqs[:num_ques]  # Ensure we only take num_ques questions
                    }
                    # Write the report data to file
                    json.dump(report_data, file)
                    file.write(',\n' if i < len(reports) - 1 else '\n')
                    file.flush()
                    total_mcqs += len(formatted_mcqs[:num_ques])
                    total_reports_processed += 1

            # Write the closing of the JSON structure
            file.write(']\n}')

        # Update metadata
        with open(output_file, 'r+') as file:
            content = file.read()
            file.seek(0)
            content = content.replace('"total_mcqs": 0', f'"total_mcqs": {total_mcqs}')
            content = content.replace(
                f'"total_reports": {len(reports)}', 
                f'"total_reports": {total_reports_processed}'
            )
            file.write(content)
            file.truncate()

        return total_reports_processed, total_mcqs
    except Exception as e:
        print(f"Error in generate_and_write_mcqs: {e}")
        return 0, 0

def swap_answer_choices(question, rng):
    """Swap the correct answer with another choice in a question."""
    # Create a deep copy of the question to avoid modifying the original
    modified_question = copy.deepcopy(question)

    # Ensure the question has the required keys
    if 'options' not in modified_question or 'correct_answer' not in modified_question:
        return modified_question

    current_correct = modified_question['correct_answer']
    other_options = [opt for opt in modified_question['options'].keys() if opt != current_correct]
    
    if other_options:
        swap_option = rng.choice(other_options)
        modified_question['options'][current_correct], modified_question['options'][swap_option] = \
            modified_question['options'][swap_option], modified_question['options'][current_correct]
        modified_question['correct_answer'] = swap_option

    return modified_question

def process_json_file(input_file, output_file, seed=123):
    """Read JSON file, swap answer choices, and save to new file."""
    # Create RNG once at the start
    rng = random.Random(seed)
    
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Track the number of changes
    total_questions = 0
    modified_questions = 0

    # Process the mcq_data
    mcq_data = data['mcq_data']

    # Iterate through each report/item in mcq_data
    for i, report in enumerate(mcq_data):
        # Swap answer choices for each question in the report
        for j, question in enumerate(report['questions']):
            total_questions += 1
            # Pass the rng instance instead of seed
            modified_question = swap_answer_choices(question, rng)
            
            # Only replace if the question was actually modified
            if modified_question != question:
                data['mcq_data'][i]['questions'][j] = modified_question
                modified_questions += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Total questions processed: {total_questions}")
    print(f"Questions modified: {modified_questions}")
    print(f"Processed file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate and Shuffle MCQs')
    parser.add_argument('--input_csv', default=os.getenv("RRGEVAL_INPUT_CSV_PATH", ""), help='Input CSV file path')
    parser.add_argument('--output_dir', default=os.getenv("RRGEVAL_OUTPUT_DIR", ""), help='Output directory')
    parser.add_argument('--reference', choices=['gt', 'gen'], default='gt', help='Reference type')
    parser.add_argument('--num_questions', type=int, default=40, help='Number of questions per report')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--use_sequential_generation', action='store_true',
        help='Generate questions in context-aware sequential batches instead of one single-shot request')
    parser.add_argument('--sequential_batch_size', type=int, default=10,
        help='Max questions requested per round in sequential generation mode (only used with --use_sequential_generation)')
    parser.add_argument('--sequential_stop_threshold', type=int, default=2,
        help='Consecutive under-attempted batches before stopping early in sequential mode (only used with --use_sequential_generation)')

    args = parser.parse_args()

    seed = args.seed
    num_ques = args.num_questions
    reference = args.reference
    input_csv = args.input_csv
    output_dir = args.output_dir
    
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv(input_csv)
    reports = df['ground_truth_report' if reference == 'gt' else 'generated_report'].tolist()
    print(f"Total unique reports: {len(reports)}")
    

    # Generate original MCQs
    output_dir = f"{output_dir}/orig_data/{reference}_reports_as_ref"
    ensure_dir(output_dir)
    json_output_file = f"{output_dir}/mcqa_data.json"
    
    # Validate config before proceeding
    if not Config.validate_config():
        print("Configuration validation failed. Please check your environment variables.")
        return

    total_reports, total_mcqs = generate_and_write_mcqs(
        reports, 
        num_ques, 
        json_output_file, 
        url=Config.API_URL,
        api_key=Config.API_KEY,
        timeout=Config.GENERATION_TIMEOUT,
        max_tokens=Config.GENERATION_MAX_TOKENS,
        temperature=Config.DEFAULT_TEMPERATURE,
        top_p=Config.DEFAULT_TOP_P,
        n=Config.DEFAULT_N,
        seed=seed,
        model_name=Config.MODEL_NAME,
        use_sequential_generation=args.use_sequential_generation,
        sequential_batch_size=args.sequential_batch_size,
        sequential_stop_threshold=args.sequential_stop_threshold
    )
    
    print(f"MCQs saved to {json_output_file}")
    print(f"Total reports processed: {total_reports}")
    print(f"Total MCQs generated: {total_mcqs}")
    
    # Generate shuffled version
    shuffled_output_dir = f"{args.output_dir}/shuffled_ans_choices_data/{args.reference}_reports_as_ref"
    ensure_dir(shuffled_output_dir)
    shuffled_output_file = f"{shuffled_output_dir}/mcqa_data.json"
    
    # Process and save shuffled version
    process_json_file(json_output_file, shuffled_output_file, seed=seed)

if __name__ == "__main__":
    main()