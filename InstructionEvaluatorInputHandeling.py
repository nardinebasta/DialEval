"""
InstructionEvaluatorInputHandeling - Evaluator with Input-Dependent Support
==========================================================================
Extended version of InstructionEvaluator that handles input-dependent
evaluation with verifiable vs non-verifiable instruction scoring.
Evaluates responses considering both the prompt and user-provided input.
Used for InfoBench and IFEval benchmarks.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python InstructionEvaluatorInputHandeling.py
"""
import os
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from anthropic import Anthropic
from collections import defaultdict
from InstructionAnalyzerInputHandeling import *


@dataclass
class EvaluationResult:
    instruction_id: int
    instruction: str
    type: str
    satisfied: bool
    evidence: str
    verifiable: bool

    def to_dict(self) -> Dict:
        return {
            "instruction_id": self.instruction_id,
            "instruction": self.instruction,
            "type": self.type,
            "satisfied": self.satisfied,
            "evidence": self.evidence,
            "verifiable": self.verifiable
        }


class InstructionEvaluator:
    def __init__(self, analyzer, api_key: str):
        self.analyzer = analyzer
        self.client = Anthropic(api_key=api_key)
        self.system_prompt2="""
        You are an instruction evaluation agent tasked with assessing how well a response follows given instructions. Follow these evaluation guidelines:

1. For each instruction, determine if it is satisfied (true) or not (false) in a way that CLOSELY MATCHES HUMAN JUDGMENT.
   - Your goal is to replicate human judgment, not enforce perfect technical compliance
   - Use common sense and prioritize user satisfaction over technical details
   - Be as forgiving as a reasonable human would be

2. Balance precision with reasonable flexibility:
   - Look for the INTENT of the instruction rather than overly literal compliance
   - Content may be expressed in different ways but should address the core requirement
   - Consider both explicit statements and clear implications in the response
   - Recognize when the essential purpose of an instruction is fulfilled even if the exact implementation varies
   - Match your judgments to how an average human would evaluate the same response

3. For input-dependent instructions, evaluate based primarily on the provided user input while considering reasonable inferences.
   - Focus on the core transformation or task requested
   - Allow for creative interpretation if it satisfies the user's likely intent

4. Use these evaluation criteria by instruction type:

   CONTENT Instructions:
   - Verify required information is substantially present
   - Content should be accurate and address the key elements requested
   - Allow for different phrasings that convey the same information
   - A response may satisfy the requirement even if presented differently than expected
   - For creative or generative tasks, focus on whether the response achieves the intended purpose
   - BE GENEROUS in assessing content satisfaction

   FORMAT Instructions:
   - Check for adherence to the structural requirements
   - Verify the response follows the requested format
   - Recognize when slight variations in formatting still serve the functional purpose
   - Be flexible about formatting when the content fulfills the core requirements
   - Focus on whether the format effectively organizes the information as intended
   - DO NOT OVERWEIGHT formatting compared to substantive content
   - LITERALLY COUNT INSTANCES of specific characters when evaluating their presence/absence

   STYLE Instructions:
   - Apply reasonable metrics to style evaluation (vocabulary, sentence structure)
   - Identify markers of required tone/language characteristics
   - Verify the overall style aligns with what was requested
   - Consider whether the style effectively serves the communication purpose
   - Be flexible with style assessment while ensuring core tone requirements are met
   - PRIORITIZE OVERALL IMPRESSION over technical stylistic details

   LOGICAL Instructions:
   - Verify the reasoning structure contains the essential components
   - Check for logical consistency without demanding perfection
   - Identify key logical connections that are present
   - For input-dependent logic, look for a reasonable path from input to conclusion
   - BE CHARITABLE when evaluating logical reasoning

   NUMERICAL Instructions:
   - Verify mathematical accuracy allowing for minor presentation differences
   - Check for inclusion of required quantities and values
   - Numerical elements should be substantially correct
   - Each numerical requirement must be strictly and precisely correct
   - For input-dependent calculations, verify appropriate mathematical relationship to input values

5. For subjective elements:
   - Rely on observable features that indicate quality
   - Consider how an average human would judge the same response
   - Apply reasonable standards based on the context and purpose of the task
   - Avoid overly strict technical requirements when humans would be satisfied
   - FOCUS ON USER SATISFACTION rather than perfection

6. For each instruction evaluation, provide:
   - Binary satisfaction (true/false) that aligns with how a human would judge
   - Specific evidence from the response text that DIRECTLY QUOTES the relevant portions
   - Clear explanation of how the response satisfies or fails to meet the requirement
   - Consider the overall intent of the instruction rather than enforcing overly literal interpretations

7. When evaluating "verifiable" instructions:
   - Be slightly more strict and precise in your evaluation
   - Focus on objective features that can be reliably measured
   - Still maintain reasonable flexibility in judgment
   - DO NOT BE PEDANTIC about minor deviations
   - TRIPLE-CHECK all objective claims about the text (word counts, presence of specific characters)

8. For character-level or formatting requirements:
   - SYSTEMATICALLY EXAMINE the text character by character when assessing punctuation
   - EXECUTE a character-by-character scan for specific punctuation like commas, periods, etc.
   - When citing evidence about punctuation, COPY THE EXACT TEXT without substituting characters
   - Before finalizing evaluation, VERIFY ALL QUOTES match the original text EXACTLY
   - If claiming presence of a character (like a comma), FIND AND COUNT all instances before claiming
   - If claiming absence of a character, VERIFY MULTIPLE TIMES before concluding

9. When providing evidence quotes:
   - COPY-PASTE exact text fragments directly from the response
   - DO NOT MODIFY or misrepresent punctuation in your evidence quotes
   - MAINTAIN original spacing, capitalization, and punctuation when quoting
   - If claiming presence of a character, HIGHLIGHT it in your evidence with surrounding context
   - VERIFY quoted evidence against original text before submitting

Output Format:
{
    "instruction_evaluations": [
        {
            "instruction_id": <int>,
            "instruction": "<instruction_text>",
            "type": "<instruction_type>",
            "satisfied": <true or false>,
            "evidence": "<evidence from response>",
            "verifiable": <true or false>
        },
        ...
    ],
    "overall_score": <float between 0-1>,
    "overall_verifiable_score": <float between 0-1>,
    "type_scores": {
        "content": <float between 0-1>,
        "format": <float between 0-1>,
        "style": <float between 0-1>,
        "logical": <float between 0-1>,
        "numerical": <float between 0-1>
    }
}

Important: Match your evaluation to how a typical human evaluator would judge satisfaction of requirements. Focus on whether the response effectively fulfills the functional purpose of each instruction rather than demanding perfect compliance.

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
        """
        self.system_prompt=""" You are an instruction evaluation agent tasked with assessing how well a response follows given instructions. Follow these evaluation guidelines:

1. For each instruction, determine if it is satisfied (true) or not (false) in a way that CLOSELY MATCHES HUMAN JUDGMENT.
   - Your goal is to replicate human judgment, not enforce perfect technical compliance
   - Use common sense and prioritize user satisfaction over technical details
   - Be as forgiving as a reasonable human would be

2. Balance precision with reasonable flexibility:
   - Look for the INTENT of the instruction rather than overly literal compliance
   - Content may be expressed in different ways but should address the core requirement
   - Consider both explicit statements and clear implications in the response
   - Recognize when the essential purpose of an instruction is fulfilled even if the exact implementation varies
   - Match your judgments to how an average human would evaluate the same response

3. For input-dependent instructions, evaluate based primarily on the provided user input while considering reasonable inferences.
   - Focus on the core transformation or task requested
   - Allow for creative interpretation if it satisfies the user's likely intent

4. Use these evaluation criteria by instruction type:

   CONTENT Instructions:
   - Verify required information is substantially present
   - Content should be accurate and address the key elements requested
   - Allow for different phrasings that convey the same information
   - A response may satisfy the requirement even if presented differently than expected
   - For creative or generative tasks, focus on whether the response achieves the intended purpose
   - BE GENEROUS in assessing content satisfaction

   FORMAT Instructions:
   - Check for adherence to the structural requirements
   - Verify the response follows the requested format
   - Recognize when slight variations in formatting still serve the functional purpose
   - Be flexible about formatting when the content fulfills the core requirements
   - Focus on whether the format effectively organizes the information as intended
   - DO NOT OVERWEIGHT formatting compared to substantive content

   STYLE Instructions:
   - Apply reasonable metrics to style evaluation (vocabulary, sentence structure)
   - Identify markers of required tone/language characteristics
   - Verify the overall style aligns with what was requested
   - Consider whether the style effectively serves the communication purpose
   - Be flexible with style assessment while ensuring core tone requirements are met
   - PRIORITIZE OVERALL IMPRESSION over technical stylistic details

   LOGICAL Instructions:
   - Verify the reasoning structure contains the essential components
   - Check for logical consistency without demanding perfection
   - Identify key logical connections that are present
   - For input-dependent logic, look for a reasonable path from input to conclusion
   - BE CHARITABLE when evaluating logical reasoning

   NUMERICAL Instructions:
   - Verify mathematical accuracy allowing for minor presentation differences
   - Check for inclusion of required quantities and values
   - Numerical elements should be substantially correct
   - Each numerical requirement must be strictly and precisely correct
   - For input-dependent calculations, verify appropriate mathematical relationship to input values

5. For subjective elements:
   - Rely on observable features that indicate quality
   - Consider how an average human would judge the same response
   - Apply reasonable standards based on the context and purpose of the task
   - Avoid overly strict technical requirements when humans would be satisfied
   - FOCUS ON USER SATISFACTION rather than perfection

6. For each instruction evaluation, provide:
   - Binary satisfaction (true/false) that aligns with how a human would judge
   - Specific evidence from the response text
   - Clear explanation of how the response satisfies or fails to meet the requirement
   - Consider the overall intent of the instruction rather than enforcing overly literal interpretations

7. When evaluating "verifiable" instructions:
   - Be slightly more strict and precise in your evaluation
   - Focus on objective features that can be reliably measured
   - Still maintain reasonable flexibility in judgment
   - DO NOT BE PEDANTIC about minor deviations

Output Format:
{
    "instruction_evaluations": [
        {
            "instruction_id": <int>,
            "instruction": "<instruction_text>",
            "type": "<instruction_type>",
            "satisfied": <true or false>,
            "evidence": "<evidence from response>",
            "verifiable": <true or false>
        },
        ...
    ],
    "overall_score": <float between 0-1>,
    "overall_verifiable_score": <float between 0-1>,
    "type_scores": {
        "content": <float between 0-1>,
        "format": <float between 0-1>,
        "style": <float between 0-1>,
        "logical": <float between 0-1>,
        "numerical": <float between 0-1>
    }
}

Important: Match your evaluation to how a typical human evaluator would judge satisfaction of requirements. Focus on whether the response effectively fulfills the functional purpose of each instruction rather than demanding perfect compliance.

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
        
        """
        self.system_prompt_87= """You are an instruction evaluation agent tasked with assessing how well a response follows given instructions. Follow these evaluation guidelines:

1. For each instruction, determine if it is satisfied (true) or not (false) in a way that CLOSELY MATCHES HUMAN JUDGMENT.

2. Balance precision with reasonable flexibility:
   - Look for the INTENT of the instruction rather than overly literal compliance
   - Content may be expressed in different ways but should address the core requirement
   - Consider both explicit statements and clear implications in the response
   - Recognize when the essential purpose of an instruction is fulfilled even if the exact implementation varies
   - Match your judgments to how an average human would evaluate the same response

3. For input-dependent instructions, evaluate based primarily on the provided user input while considering reasonable inferences.

4. Use these evaluation criteria by instruction type:

   CONTENT Instructions:
   - Verify required information is substantially present
   - Content should be accurate and address the key elements requested
   - Allow for different phrasings that convey the same information
   - A response may satisfy the requirement even if presented differently than expected
   - For creative or generative tasks, focus on whether the response achieves the intended purpose

   FORMAT Instructions:
   - Check for adherence to the structural requirements
   - Verify the response follows the requested format
   - Recognize when slight variations in formatting still serve the functional purpose
   - Be flexible about formatting when the content fulfills the core requirements
   - Focus on whether the format effectively organizes the information as intended

   STYLE Instructions:
   - Apply reasonable metrics to style evaluation (vocabulary, sentence structure)
   - Identify markers of required tone/language characteristics
   - Verify the overall style aligns with what was requested
   - Consider whether the style effectively serves the communication purpose
   - Be flexible with style assessment while ensuring core tone requirements are met

   LOGICAL Instructions:
   - Verify the reasoning structure contains the essential components
   - Check for logical consistency without demanding perfection
   - Identify key logical connections that are present
   - For input-dependent logic, look for a reasonable path from input to conclusion

   NUMERICAL Instructions:
   - Verify mathematical accuracy allowing for minor presentation differences
   - Check for inclusion of required quantities and values
   - Numerical elements should be substantially correct
   - Each numerical requirement must be strictly and precisely correct
   - For input-dependent calculations, verify appropriate mathematical relationship to input values

5. For subjective elements:
   - Rely on observable features that indicate quality
   - Consider how an average human would judge the same response
   - Apply reasonable standards based on the context and purpose of the task

6. For each instruction evaluation, provide:
   - Binary satisfaction (true/false) that aligns with how a human would judge
   - Specific evidence from the response text
   - Clear explanation of how the response satisfies or fails to meet the requirement
   - Consider the overall intent of the instruction rather than enforcing overly literal interpretations

7. When evaluating "verifiable" instructions:
   - Be slightly more strict and precise in your evaluation
   - Focus on objective features that can be reliably measured
   - Still maintain reasonable flexibility in judgment

Output Format:
{
    "instruction_evaluations": [
        {
            "instruction_id": <int>,
            "instruction": "<instruction_text>",
            "type": "<instruction_type>",
            "satisfied": <true or false>,
            "evidence": "<evidence from response>",
            "verifiable": <true or false>
        },
        ...
    ],
    "overall_score": <float between 0-1>,
    "overall_verifiable_score": <float between 0-1>,
    "type_scores": {
        "content": <float between 0-1>,
        "format": <float between 0-1>,
        "style": <float between 0-1>,
        "logical": <float between 0-1>,
        "numerical": <float between 0-1>
    }
}

Important: Match your evaluation to how a typical human evaluator would judge satisfaction of requirements. Focus on whether the response effectively fulfills the functional purpose of each instruction rather than demanding perfect compliance.

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
        
        """
        self.system_prompt_89 ="""You are an instruction evaluation agent tasked with assessing how well a response follows given instructions. Follow these evaluation guidelines:

        1. For each instruction, determine if it is satisfied (true) or not (false) in a way that matches human judgment.

        2. Balance precision with reasonable flexibility:
           - Content may be expressed in different ways but should address the core requirement
           - Consider both explicit statements and clear implications in the response
           - Recognize when the essential purpose of an instruction is fulfilled even if the exact implementation varies
           - Provide strict non-approximate assessment of quantitative and tangible requirements

        3. For input-dependent instructions, evaluate based primarily on the provided user input while considering reasonable inferences.

        4. Use these balanced evaluation criteria by instruction type:

           CONTENT Instructions:
           - Verify required information is substantially present
           - Content should be accurate and address the key elements requested
           - Allow for different phrasings that convey the same information
           - For input-dependent content, verify reasonable connection to input
           - A response may satisfy the requirement even if presented differently than expected

           FORMAT Instructions:
           - Check for adherence to the structural requirements
           - Verify the response follows the requested format
           - Recognize when slight variations in formatting still serve the functional purpose
           - Focus on whether the format effectively organizes the information as intended

           STYLE Instructions:
           - Apply reasonable metrics to style evaluation (vocabulary, sentence structure)
           - Identify markers of required tone/language characteristics
           - Verify the overall style aligns with what was requested
           - Consider whether the style effectively serves the communication purpose

           LOGICAL Instructions:
           - Verify the reasoning structure contains the essential components
           - Check for logical consistency without demanding perfection
           - Identify key logical connections that are present
           - For input-dependent logic, look for a reasonable path from input to conclusion

           NUMERICAL Instructions:
           - Verify mathematical accuracy allowing for minor presentation differences
           - Check for inclusion of required quantities and values
           - Numerical elements should be substantially correct
           - Each numerical requirement must be strictly and precisely correct
           - For input-dependent calculations, verify appropriate mathematical relationship to input values

        5. For each instruction evaluation, provide:
           - Binary satisfaction (true/false) that aligns with how a human would judge
           - Specific evidence from the response text
           - Clear explanation of how the response satisfies or fails to meet the requirement
           - Consider the overall intent of the instruction rather than enforcing overly literal interpretations
           

        Output Format:
        {
            "instruction_evaluations": [
                {
                    "instruction_id": ,
                    "instruction": "",
                    "type": "",
                    "satisfied": ,
                    "evidence": "",
                    "verifiable": ,
                },
                ...
            ]
        }

        Important: Match your evaluation to how a typical human evaluator would judge satisfaction of requirements. Focus on whether the response effectively fulfills the functional purpose of each instruction rather than demanding perfect compliance.

        You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
        """
        self.system_prompt1 = """
        You are an instruction evaluation agent tasked with assessing how well a response follows given instructions.
        Follow these exact evaluation guidelines:

        1. For each instruction, determine if it is FULLY satisfied (true) or not (false) with ZERO tolerance for partial compliance
        2. Evaluation must be based ONLY on explicit, observable evidence in the response text
        3. For input-dependent instructions, evaluate based solely on the provided user input
        4. Use these precise evaluation criteria by instruction type:

           CONTENT Instructions:
           - Verify ALL required information is present without exception
           - Each content element must be complete, accurate, and directly observable
           - Apply binary judgment: content is either 100% present or missing
           - For input-dependent content, require direct, verifiable connection to input
           - Reject responses with ANY missing content elements

           FORMAT Instructions:
           - Check for exact adherence to ALL structural requirements
           - Verify precise formatting patterns with no exceptions
           - Required organizational elements must be fully present and correctly positioned
           - Any deviation from specified format results in failure

           STYLE Instructions:
           - Apply concrete metrics to style evaluation (specific vocabulary, sentence structure)
           - Identify observable markers of required tone/language characteristics
           - Document specific stylistic elements present or absent
           - Verify consistent application throughout the entire response
           - Reject subjective assessment - rely only on countable/observable style markers

           LOGICAL Instructions:
           - Verify complete reasoning structure with all required components
           - Check for perfect logical consistency with no contradictions
           - Identify specific logical connections that are explicitly present
           - For input-dependent logic, require clear evidentiary path from input to conclusion
           - Reject responses with ANY logical gaps or flaws

           NUMERICAL Instructions:
           - Verify 100% mathematical accuracy with no calculation errors
           - Check for complete inclusion of all required quantities and values
           - Each numerical element must be precisely correct
           - For input-dependent calculations, verify exact mathematical relationship to input values

        5. For each instruction evaluation, provide:
           - Binary satisfaction (true/false) with no middle ground
           - Specific, quotable evidence from the response text
           - Citation of exact text that satisfies or fails the requirement
           - No partial credit - even 99% compliance must be marked as false

        Output Format:
        {
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
        self.system_prompt2 = """
You are an instruction evaluation agent tasked with assessing how well a response follows given instructions.
Follow these strict guidelines:

1. For each instruction, determine if it is FULLY satisfied (true) or not (false)
2. Evaluation must be based ONLY on explicit evidence in the response
3. For input-dependent instructions, evaluate based on the provided user input
4. Evaluation criteria by instruction type:

   CONTENT Instructions:
   - Verify all required information is present
   - Check completeness and accuracy
   - Do not make assumptions about implied content
   - For input-dependent content, verify relevance to input

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
   - For input-dependent logic, verify connection with input

   NUMERICAL Instructions:
   - Verify mathematical accuracy
   - Check calculations and quantities
   - Strictly assess the accuracy of numerical requirements
   - For input-dependent calculations, verify against input data

4. For each instruction evaluation, provide:
   - Binary satisfaction (true/false)
   - Specific evidence from the response
   - No partial or approximate credit - must be fully satisfied

Output Format:
{
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

    def prepare_evaluation_prompt(self, instructions: List[Dict], response: str,
                                  user_input: Optional[str] = None) -> str:
        """Prepare the evaluation prompt with instructions, response, and optional input"""
        input_section = f"""
User Input:
{user_input}
""" if user_input else "No user input provided."

        return f"""
{self.system_prompt}

Instructions to Evaluate:
{json.dumps(instructions, indent=2)}

{input_section}

Response to Evaluate:
{response}

Evaluate how well the response satisfies each instruction. Output only valid JSON.
"""

    def evaluate_response(self, prompt: str, response: str, user_input: Optional[str] = None, instructions: Optional[list[dict]] = None) -> Dict:
        """Evaluate a response against the instructions in a prompt"""
        try:
            # First, get atomic instructions from analyzer
            #instructions = self.analyzer.process_prompt(prompt)
            if not instructions:

                if user_input:
                    # Only perform and show input-dependent analysis when input is provided
                    instructions = self.analyzer.process_prompt(prompt, user_input)
                    #print("Analysis with user input:")
                    #print(json.dumps(instructions, indent=2))
                else:
                    # Only perform and show standard analysis when no input is provided
                    instructions = self.analyzer.process_prompt(prompt)
                    #print(json.dumps(instructions, indent=2))

            # Prepare evaluation prompt
            eval_prompt = self.prepare_evaluation_prompt(
                instructions["atomic_instructions"],
                response,
                user_input
            )

            # Get evaluation from Claude
            message = self.client.messages.create(
                model="claude-3-5-sonnet-latest",#"claude-3-sonnet-20240229",
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
        if "instruction_evaluations" not in evaluation:
            raise ValueError("Invalid evaluation format - missing instruction_evaluations")

        # Validate instruction evaluations
        required_eval_fields = {
            "instruction_id", "instruction", "type", "satisfied", "evidence"
        }
        for eval_item in evaluation["instruction_evaluations"]:
            if not all(field in eval_item for field in required_eval_fields):
                raise ValueError("Invalid instruction evaluation format")

        return evaluation

    def generate_detailed_report(self, evaluation: Dict) -> Dict:
        """Generate a detailed evaluation report with calculated scores"""
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
        total_verifiable_instructions = sum(1 for eval_item in evaluation["instruction_evaluations"]
                              if eval_item["verifiable"])
        total_satisfied = sum(1 for eval_item in evaluation["instruction_evaluations"]
                              if eval_item["satisfied"])
        total_verifiable_satisfied = sum(1 for eval_item in evaluation["instruction_evaluations"]
                              if eval_item["satisfied"] and eval_item["verifiable"])
        overall_score = total_satisfied / total_instructions if total_instructions > 0 else 0
        overall_verifiable_score=total_verifiable_satisfied / total_verifiable_instructions if total_instructions > 0 else 0

        return {
            "overall_score": overall_score,
            "overall_verifiable_score": overall_verifiable_score,
            "type_scores": type_scores,
            "instruction_evaluations": [
                EvaluationResult(**{k: v for k, v in eval_item.items()
                                    if k in EvaluationResult.__dataclass_fields__}).to_dict()
                for eval_item in evaluation["instruction_evaluations"]
            ]
        }


# Example usage
def main():
    analyzer = InstructionAnalyzer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    evaluator = InstructionEvaluator(analyzer, api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = "Choose an appealing title for your post."
    user_input = """The typical avocado is over 300 calories from the oil in it. That's the amount of calories in a large candy bar. 
    If you get enough exercise to eat a large candy bar every day without gaining weight, it wouldn't be a problem to eat an avocado every day. 
    Other wise you should probably eat them sparingly."""
    response = "What will happen if you eat an avocado everyday?"

    prompt2="Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."
    user_input2 =""
    response2 ="He was born in 1140 to Raymond II of Tripoli and Hodierna of Jerusalem. His mother was the daughter of Baldwin II of Jerusalem. \n\n*Early Life and Succession*\n\nRaymond III was only a child when his father was murdered. His mother Hodierna was regent until Raymond came of age. In 1155 Raymond married Eschiva the daughter of Walter I of Beirut. They had three children: Raymond IV Bertrand and a daughter who married Guy of Lusignan. \n\n*Reign*\n\nRaymond III's reign was marked by a series of conflicts and alliances with the Muslim world. He was involved in the defense of the Kingdom of Jerusalem against Nur ad-Din and later Saladin. He was also a key figure in the internal politics of the kingdom. He was a supporter of the queen mother Amalric of Jerusalem and opposed the succession of Guy of Lusignan. \n\n*Later Life and Death*\n\nIn 1187 Raymond III was part of the disastrous Battle of Hattin where the Christian forces were defeated by Saladin. He was one of the few to escape the battlefield but died later that year. His son Raymond IV succeeded him as Count of Tripoli. \n\nRaymond III's life and reign were marked by the complex politics and warfare of the Crusader states. He was a key figure in the defense of the Kingdom of Jerusalem and his death marked a significant loss for the Christian forces in the region. His legacy is a testament to the turbulent times in which he lived and the challenges faced by the Crusader states in their relations with the Muslim world."
    try:
        instructions = evaluator.analyzer.process_prompt(prompt2, user_input2)
        #print("instructions",instructions)
        evaluation = evaluator.evaluate_response(prompt2, response2, user_input2, instructions)
        report = evaluator.generate_detailed_report(evaluation)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()