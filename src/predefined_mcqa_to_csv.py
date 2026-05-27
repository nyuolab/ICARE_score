"""
Convert a global predefined question bank (questions.json) to a flat CSV
with one row per question. Report_ID is not included — the evaluation script
applies these questions to every report in the evaluation set.
"""
import json
import csv
import argparse
import os


def convert(input_json: str, output_csv: str) -> int:
    with open(input_json, 'r') as f:
        data = json.load(f)

    questions = data['questions']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    headers = ['Question_ID', 'Category', 'Question_Text', 'Options', 'Correct_Answer']
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for q in questions:
            writer.writerow({
                'Question_ID': q['question_id'],
                'Category': q.get('category', ''),
                'Question_Text': q['question_text'],
                'Options': str(q['options']),
                'Correct_Answer': q['correct_answer'],
            })

    print(f"Wrote {len(questions)} questions to {output_csv}")
    return len(questions)


def main():
    parser = argparse.ArgumentParser(
        description='Convert predefined question bank JSON to evaluation CSV'
    )
    parser.add_argument('--input_json', required=True,
                        help='Path to questions.json (global question bank)')
    parser.add_argument('--output_csv', required=True,
                        help='Path for output CSV')
    args = parser.parse_args()
    convert(args.input_json, args.output_csv)


if __name__ == '__main__':
    main()
