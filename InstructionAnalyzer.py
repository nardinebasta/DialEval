"""
InstructionAnalyzer - Atomic Instruction Decomposition Agent
==========================================================
Decomposes complex prompts into atomic, independently evaluatable
instructions classified by type (content, format, style, logical,
numerical). First stage of the DIALEVAL Analyze -> Evaluate pipeline.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python InstructionAnalyzer.py
"""
import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
from anthropic import Anthropic


@dataclass
class AtomicInstruction:
    id: int
    text: str
    type: str
    parent_id: Optional[int] = None
    dependencies: List[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "instruction": self.text,
            "type": self.type,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies or []
        }


class InstructionAnalyzer:
    INSTRUCTION_TYPES = [
        "content",  # Information/content requirements
        "format",  # Structural/formatting requirements
        "style",  # Writing style/tone requirements
        "logical",  # Reasoning/logic requirements
        "numerical"  # Mathematical/quantitative requirements
    ]

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = """
You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions.
Follow these strict guidelines:

1. Analyze the given prompt to identify ALL explicit and implicit instructions/requirements
2. Break down compound instructions into atomic (indivisible) components
3. Classify each atomic instruction into one of these types:
   - content: Information/content requirements
   - format: Structural/formatting requirements
   - style: Writing style/tone requirements
   - logical: Reasoning/logic requirements
   - numerical: Mathematical/quantitative requirements
4. Identify dependencies between instructions
5. Ensure NO information is lost, added, or modified from the original prompt
6. Each atomic instruction should be:
   - Independent: Can be evaluated separately
   - Unambiguous: Clear success criteria
   - Atomic: Cannot be broken down further
   - Preserved: Maintains original intent

Output Format:
{
    "atomic_instructions": [
        {
            "id": <int>,
            "instruction": "<clear, evaluatable instruction>",
            "type": "<instruction_type>",
            "parent_id": <int or null>,
            "dependencies": [<list of instruction IDs>]
        },
        ...
    ]
}

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
"""

    def prepare_input_prompt(self, prompt: str) -> str:
        return f"""
{self.system_prompt}

Input Prompt:
{prompt}

Extract and analyze all instructions from this prompt. Maintain original requirements without adding, removing, or modifying constraints. Output only valid JSON.
"""

    def validate_instruction_types(self, instructions: List[Dict]) -> bool:
        """Validate that all instruction types are valid"""
        return all(instr["type"] in self.INSTRUCTION_TYPES for instr in instructions)

    def validate_dependencies(self, instructions: List[Dict]) -> bool:
        """Validate that all referenced dependencies exist"""
        ids = {instr["id"] for instr in instructions}
        for instr in instructions:
            if not all(dep in ids for dep in instr.get("dependencies", [])):
                return False
            if instr.get("parent_id") and instr["parent_id"] not in ids:
                return False
        return True

    def validate_output(self, output: Dict) -> bool:
        """Validate the structure and content of the analyzer output"""
        if not isinstance(output, dict) or "atomic_instructions" not in output:
            return False

        instructions = output["atomic_instructions"]
        if not isinstance(instructions, list):
            return False

        required_fields = {"id", "instruction", "type"}
        if not all(isinstance(instr, dict) and required_fields.issubset(instr.keys())
                   for instr in instructions):
            return False

        return (self.validate_instruction_types(instructions) and
                self.validate_dependencies(instructions))

    def extract_atomic_instructions(self, prompt: str) -> Dict:
        """Process the prompt using Claude 4.5 Sonnet to extract atomic instructions"""
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Extract JSON from response
            response_text = message.content[0].text
            try:
                output = json.loads(response_text)
            except json.JSONDecodeError:
                # Strip markdown code fences if present (common with newer models)
                cleaned = re.sub(r'```(?:json)?\s*', '', response_text).strip()
                cleaned = re.sub(r'```\s*$', '', cleaned).strip()
                try:
                    output = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Try to extract the outermost JSON object (supporting nested braces)
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if json_match:
                        output = json.loads(json_match.group())
                    else:
                        raise ValueError("Could not extract valid JSON from response")

            return output

        except Exception as e:
            raise Exception(f"Error calling Claude API: {str(e)}")

    def process_prompt(self, prompt: str) -> Dict:
        """Main method to process a prompt and extract atomic instructions"""
        try:
            input_prompt = self.prepare_input_prompt(prompt)
            output = self.extract_atomic_instructions(input_prompt)
            print(output)

            if not self.validate_output(output):
                raise ValueError("Invalid output structure from instruction analysis")

            return output

        except Exception as e:
            raise Exception(f"Error processing prompt: {str(e)}")


# Example usage:
def main():
    # Initialize with your API key
    analyzer = InstructionAnalyzer(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    sample_prompt = """
    Write a comprehensive historical analysis of World War II. 
    Include key dates, major battles, and significant figures. 
    Format the response in chronological order using proper headings.
    Ensure all claims are supported by evidence.
    Keep the tone academic and formal.
    Calculate the total casualties across all major battles.
    """

    try:
        result = analyzer.process_prompt(sample_prompt)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()


