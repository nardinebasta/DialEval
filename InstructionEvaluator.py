"""
InstructionEvaluator - Response Evaluation Agent
================================================
Evaluates LLM responses against atomic instructions produced by
InstructionAnalyzer. Outputs per-instruction binary satisfaction
judgments with evidence and aggregated per-type / overall scores.
Second stage of the DIALEVAL Analyze -> Evaluate pipeline.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python InstructionEvaluator.py
"""
import os
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from anthropic import Anthropic
from collections import defaultdict
from InstructionAnalyzer import *


@dataclass
class EvaluationResult:
    instruction_id: int
    instruction: str
    type: str
    satisfied: bool
    evidence: str

    def to_dict(self) -> Dict:
        return {
            "instruction_id": self.instruction_id,
            "instruction": self.instruction,
            "type": self.type,
            "satisfied": self.satisfied,
            "evidence": self.evidence
        }


class InstructionEvaluator:
    def __init__(self, analyzer, api_key: str):
        self.analyzer = analyzer
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = """
You are an instruction evaluation agent tasked with assessing how well a response follows given instructions.
Follow these strict guidelines:

1. For each instruction, determine if it is FULLY satisfied (true) or not (false)
2. Evaluation must be based ONLY on explicit evidence in the response
3. Evaluation criteria by instruction type:

   CONTENT Instructions:
   - Verify all required information is present
   - Check completeness and accuracy
   - Do not make assumptions about implied content

   FORMAT Instructions:
   - Check exact structural requirements
   - Verify formatting patterns
   - Ensure organizational elements are present

   STYLE Instructions:
   - Verify tone and language characteristics
   - Check writing style consistency
   - Assess adherence to stylistic requirements

   LOGICAL Instructions:
   - Verify reasoning structure
   - Check argument consistency
   - Assess logical flow and connections

   NUMERICAL Instructions:
   - Verify mathematical accuracy
   - Check calculations and quantities
   - Assess numerical requirements

4. For each instruction evaluation, provide:
   - Binary satisfaction (true/false)
   - Specific evidence from the response
   - No partial credit - must be fully satisfied

Output Format:
{
    "overall_score": <float>,
    "type_scores": {
        "content": <float>,
        "format": <float>,
        "style": <float>,
        "logical": <float>,
        "numerical": <float>
    },
    "instruction_evaluations": [
        {
            "instruction_id": <int>,
            "instruction": "<instruction_text>",
            "type": "<instruction_type>",
            "satisfied": <boolean>,
            "evidence": "<specific_evidence_from_response>"
        },
        ...
    ]
}

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
"""

    def prepare_evaluation_prompt(self, instructions: List[Dict], response: str) -> str:
        """Prepare the evaluation prompt with instructions and response"""
        return f"""
{self.system_prompt}

Instructions to Evaluate:
{json.dumps(instructions, indent=2)}

Response to Evaluate:
{response}

Evaluate how well the response satisfies each instruction. Output only valid JSON.
"""

    def evaluate_response(self, prompt: str, response: str) -> Dict:
        """Evaluate a response against the instructions in a prompt"""
        try:
            # First, get atomic instructions from analyzer
            instructions = self.analyzer.process_prompt(prompt)

            # Prepare evaluation prompt
            eval_prompt = self.prepare_evaluation_prompt(
                instructions["atomic_instructions"],
                response
            )

            # Get evaluation from Claude
            message = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": eval_prompt
                    }
                ]
            )

            # Parse evaluation results
            response_text = message.content[0].text
            try:
                evaluation = json.loads(response_text)
            except json.JSONDecodeError:
                # Strip markdown code fences if present (common with newer models)
                cleaned = re.sub(r'```(?:json)?\s*', '', response_text).strip()
                cleaned = re.sub(r'```\s*$', '', cleaned).strip()
                try:
                    evaluation = json.loads(cleaned)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if json_match:
                        evaluation = json.loads(json_match.group())
                    else:
                        raise ValueError("Could not extract valid JSON from evaluation response")

            # Validate and process results
            return self.process_evaluation_results(evaluation)

        except Exception as e:
            raise Exception(f"Error in evaluation process: {str(e)}")

    def process_evaluation_results(self, evaluation: Dict) -> Dict:
        """Process and validate evaluation results"""
        required_fields = {"overall_score", "type_scores", "instruction_evaluations"}
        if not all(field in evaluation for field in required_fields):
            raise ValueError("Invalid evaluation format - missing required fields")

        # Validate score ranges
        if not (0 <= evaluation["overall_score"] <= 1):
            raise ValueError("Overall score must be between 0 and 1")

        for type_score in evaluation["type_scores"].values():
            if not (0 <= type_score <= 1):
                raise ValueError("Type scores must be between 0 and 1")

        # Validate instruction evaluations
        required_eval_fields = {
            "instruction_id", "instruction", "type", "satisfied", "evidence"
        }
        for eval_item in evaluation["instruction_evaluations"]:
            if not all(field in eval_item for field in required_eval_fields):
                raise ValueError("Invalid instruction evaluation format")

        return evaluation

    def generate_detailed_report(self, evaluation: Dict) -> Dict:
        """Generate a detailed evaluation report"""
        # Count instructions by type
        type_counts = defaultdict(int)
        type_satisfied = defaultdict(int)

        for eval_item in evaluation["instruction_evaluations"]:
            instr_type = eval_item["type"]
            type_counts[instr_type] += 1
            if eval_item["satisfied"]:
                type_satisfied[instr_type] += 1

        # Calculate type scores
        type_scores = {
            instr_type: type_satisfied[instr_type] / count
            for instr_type, count in type_counts.items()
        }

        # Calculate overall score
        total_instructions = len(evaluation["instruction_evaluations"])
        total_satisfied = sum(1 for eval_item in evaluation["instruction_evaluations"]
                              if eval_item["satisfied"])
        overall_score = total_satisfied / total_instructions if total_instructions > 0 else 0

        return {
            "overall_score": overall_score,
            "type_scores": type_scores,
            "instruction_evaluations": [
                EvaluationResult(**{k: v for k, v in eval_item.items()
                                    if k in EvaluationResult.__dataclass_fields__}).to_dict()
                for eval_item in evaluation["instruction_evaluations"]
            ]
        }


# Example usage
def main():
    # Initialize with your API key
    analyzer = InstructionAnalyzer(api_key=os.environ.get("ANTHROPIC_API_KEY"))  # You'll need to import this
    evaluator = InstructionEvaluator(analyzer, api_key=os.environ.get("ANTHROPIC_API_KEY"))

    sample_prompt = """
    Write a comprehensive historical analysis of World War II. 
    Include key dates, major battles, and significant figures. 
    Format the response in chronological order using proper headings.
    Ensure all claims are supported by evidence.
    Keep the tone academic and formal.
    Calculate the total casualties across all major battles.
    """

    sample_response = """
    World War II (1939-1945) was the deadliest military conflict in history, claiming over 70 million lives. The war began with Nazi Germany's invasion of Poland on September 1, 1939, prompting Britain and France to declare war. Early German victories in Europe were followed by key turning points including the Battle of Britain (1940), Operation Barbarossa against the Soviet Union (1941), and Japan's attack on Pearl Harbor bringing the US into the war (December 7, 1941). Major Allied victories at Stalingrad (1942-43), El Alamein (1942), and Midway (1942) shifted momentum. The D-Day invasion (June 6, 1944) opened a Western Front while Soviet forces advanced from the East. The war in Europe ended with Germany's surrender on May 7, 1945, followed by Japan's surrender on August 15, 1945, after atomic bombs were dropped on Hiroshima and Nagasaki. The conflict reshaped global politics, establishing the US and USSR as superpowers and leading to the formation of the United Nations.
    """

    try:
        evaluation = evaluator.evaluate_response(sample_prompt, sample_response)
        report = evaluator.generate_detailed_report(evaluation)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()