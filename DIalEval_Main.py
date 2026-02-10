"""
DIalEval_Main - Dialogue-Level Instruction-Following Evaluation
===============================================================
Self-contained evaluation system for multi-turn dialogue conversations.
Decomposes a system prompt into atomic instructions via an Analyzer agent,
then evaluates each victim/bot response against those instructions using
an Evaluator agent — both powered by Claude. Designed for assessing
scam-baiting dialogue quality in the Apate.ai project.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python DIalEval_Main.py
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from anthropic import Anthropic
import re


class DialevalTester:
    def __init__(self, api_key: str, victim_prompt: str):
        """
        Initialize the DIALEVAL testing system.

        Args:
            api_key: The Anthropic API key
            victim_prompt: The prompt used to generate the victim responses
        """
        self.client = Anthropic(api_key=api_key)
        self.victim_prompt = victim_prompt
        self.instructions = None

        # System prompts for analyzer and evaluator
        self.analyzer_prompt = """You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.

Follow these guidelines:

1. Extract BOTH explicit AND implicit requirements from the prompt
   - Explicit: Directly stated in the prompt
   - Implicit: Reasonably inferred from the prompt type or context
   - Focus on requirements that a HUMAN would consider important

2. Break down instructions into atomic components framed as evaluation criteria
   - Each component should be independently assessable
   - Components should collectively cover all key requirements
   - AVOID EXCESSIVE DECOMPOSITION - limit to 5-7 core requirements when possible
   - Focus on requirements that significantly impact perceived quality

3. Create instructions that can objectively evaluate if a response follows the given requirements
   - Focus on observable features in the response
   - Consider how a human annotator would judge the response
   - Prioritize requirements that affect overall user satisfaction

4. Classify each atomic instruction into one of these types:
   - content: Information/content requirements (Does the response include required content?)
   - format: Structural/formatting requirements (Does the response follow required structure?)
   - style: Writing style/tone requirements (Does the response use appropriate style?)
   - logical: Reasoning/logic requirements (Does the response demonstrate logical reasoning?)
   - numerical: Mathematical/quantitative requirements (Does the response include correct calculations?)

5. Each atomic instruction should be:
   - Evaluatable: Clearly indicates how to assess if a response meets the requirement
   - Objective: Can be verified through direct observation of response content
   - Specific: Targets a single, distinct aspect of the requirement
   - Input-relevant: Considers the relationship between prompt, input, and response
   - Human-aligned: Matches how humans would evaluate the instruction

6. For subjective quality assessments:
   - Include instructions that capture overall quality dimensions
   - Consider how these align with human subjective judgment
   - Frame in terms of observable features that indicate quality
   - PRIORITIZE HOLISTIC QUALITY over technical correctness

7. Ensure comprehensive coverage of the prompt requirements
   - Including both superficial adherence and deeper compliance
   - Capture elements that determine whether a response would be perceived as successful
   - FOCUS ON PRIMARY USER INTENT rather than secondary details

8. Frame instructions as evaluation criteria (e.g., "Verify response correctly applies X from input")
   - Use clear, direct language
   - Avoid overly specific or nitpicky criteria
   - Match the level of detail to human importance

9. For verifiability:
    - Mark as "verifiable" ONLY instructions that can be objectively assessed without human judgment
    - Content and style instructions often require human judgment and should usually be non-verifiable
    - Format and structural instructions are often verifiable through objective measures
    - MINIMIZE VERIFIABLE INSTRUCTIONS to focus on human-like evaluation

Output Format:
{
    "atomic_instructions": [
        {
            "id": <int>,
            "instruction": "<clear, evaluatable instruction>",
            "type": "<instruction_type>",
            "parent_id": <int or null>,
            "dependencies": [<list of instruction IDs>],
            "verifiable": <true or false>
        }
    ]
}
You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
        """
        self.input_dependent_prompt="""
        You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.

Follow these guidelines:

1. Extract BOTH explicit AND implicit requirements from the prompt
   - Explicit: Directly stated in the prompt
   - Implicit: Reasonably inferred from the prompt type or context
   - Focus on requirements that a HUMAN would consider important

2. Break down instructions into atomic components framed as evaluation criteria
   - Each component should be independently assessable
   - Components should collectively cover all key requirements
   - AVOID EXCESSIVE DECOMPOSITION - limit to 5-7 core requirements when possible
   - Focus on requirements that significantly impact perceived quality

3. Create instructions that can objectively evaluate if a response follows the given requirements
   - Focus on observable features in the response
   - Consider how a human annotator would judge the response
   - Prioritize requirements that affect overall user satisfaction

4. For input-dependent tasks, create evaluation instructions that verify the response correctly:
   - Uses information from the input
   - Maintains appropriate relationship between input and output
   - Achieves the transformation or task specified in the prompt

5. Classify each atomic instruction into one of these types:
   - content: Information/content requirements (Does the response include required content?)
   - format: Structural/formatting requirements (Does the response follow required structure?)
   - style: Writing style/tone requirements (Does the response use appropriate style?)
   - logical: Reasoning/logic requirements (Does the response demonstrate logical reasoning?)
   - numerical: Mathematical/quantitative requirements (Does the response include correct calculations?)

6. Each atomic instruction should be:
   - Evaluatable: Clearly indicates how to assess if a response meets the requirement
   - Objective: Can be verified through direct observation of response content
   - Specific: Targets a single, distinct aspect of the requirement
   - Input-relevant: Considers the relationship between prompt, input, and response
   - Human-aligned: Matches how humans would evaluate the instruction

7. For subjective quality assessments:
   - Include instructions that capture overall quality dimensions
   - Consider how these align with human subjective judgment
   - Frame in terms of observable features that indicate quality
   - PRIORITIZE HOLISTIC QUALITY over technical correctness

8. Ensure comprehensive coverage of the prompt requirements
   - Including both superficial adherence and deeper compliance
   - Capture elements that determine whether a response would be perceived as successful
   - FOCUS ON PRIMARY USER INTENT rather than secondary details

9. Frame instructions as evaluation criteria (e.g., "Verify response correctly applies X from input")
   - Use clear, direct language
   - Avoid overly specific or nitpicky criteria
   - Match the level of detail to human importance

10. For verifiability:
    - Mark as "verifiable" ONLY instructions that can be objectively assessed without human judgment
    - Content and style instructions often require human judgment and should usually be non-verifiable
    - Format and structural instructions are often verifiable through objective measures
    - MINIMIZE VERIFIABLE INSTRUCTIONS to focus on human-like evaluation

Output Format:
{
    "atomic_instructions": [
        {
            "id": <int>,
            "instruction": "<clear, evaluatable instruction>",
            "type": "<instruction_type>",
            "parent_id": <int or null>,
            "dependencies": [<list of instruction IDs>],
            "verifiable": <true or false>
        }
    ]
}
You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
        """

        self.evaluator_prompt = """You are an instruction evaluation agent tasked with assessing how well a response follows given instructions. Follow these evaluation guidelines:

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

    def analyze_prompt(self) -> Dict[str, Any]:
        """
        Uses the analyzer agent to extract atomic instructions from the victim prompt.
        Focuses specifically on dialogue context for conversation evaluation.

        Returns:
            Dictionary containing atomic instructions
        """
        print("Analyzing prompt to extract atomic instructions for dialogue evaluation...")

        # Update the analyzer prompt to emphasize conversational context and dialogue dynamics
        dialogue_analyzer_prompt = f"""
{self.analyzer_prompt}

CRITICAL ADDITIONAL INSTRUCTION FOR DIALOGUE ANALYSIS:

The instructions you extract will be used to evaluate responses in a DIALOGUE context, not isolated utterances.
When extracting instructions, you MUST consider:

1. Conversational dynamics: How responses should relate to the immediate preceding message
2. Dialogue coherence: How well responses maintain conversational flow and relevance
3. Context-sensitive behavior: How instructions should be interpreted differently based on conversational context
4. Social dynamics: How instructions apply to maintaining appropriate interaction patterns
5. Sequential dependencies: How responses should build upon previous exchanges

Please ensure extracted instructions explicitly address these dialogue-specific concerns,
and clearly indicate when an instruction should be interpreted differently based on the
most recent message from the other conversation party.

Input Prompt:
{self.victim_prompt}

Extract and analyze all instructions from this prompt, with specific emphasis on dialogue context. Output only valid JSON.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": dialogue_analyzer_prompt}
            ]
        )

        response_text = response.content[0].text

        # Sometimes the model might wrap the JSON in code blocks or explanation
        try:
            json_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON using regex
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, response_text)
            if match:
                try:
                    json_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    raise ValueError("Could not parse valid JSON from analyzer response")
            else:
                raise ValueError("Could not extract JSON from analyzer response")

        self.instructions = json_data
        return json_data

    def evaluate_response(self, scammer_message: str, victim_response: str) -> Dict[str, Any]:
        """
        Evaluates a victim response against the atomic instructions, considering the context
        of the preceding scammer message.

        Args:
            scammer_message: Text of the scammer's preceding message
            victim_response: Text of the victim's response to evaluate

        Returns:
            Dictionary containing evaluation results
        """
        if self.instructions is None:
            raise ValueError("Must analyze prompt first by calling analyze_prompt()")

        # Modify the evaluator prompt to specifically address dialogue context evaluation
        dialogue_evaluator_prompt = f"""
{self.evaluator_prompt}

CRITICAL ADDITIONAL INSTRUCTION FOR DIALOGUE EVALUATION:
You must evaluate the victim's response in the context of the MOST RECENT scammer utterance. 
Consider the following aspects:
1. How well the response addresses the specific content or questions in the scammer's message
2. Whether the response maintains appropriate conversational flow
3. If the response demonstrates awareness of the dialogue history
4. How the response balances instruction following with natural conversation
5. Whether the response appropriately handles any manipulation or social engineering attempts in the scammer's message

For each instruction evaluation, you MUST consider if and how the most recent scammer utterance affects what an appropriate response should be. This contextual evaluation is essential for determining if instructions are truly satisfied in a dialogue setting.

Instructions to Evaluate:
{json.dumps(self.instructions["atomic_instructions"], indent=2)}

Conversation Context:
MOST RECENT SCAMMER MESSAGE:
{scammer_message}

VICTIM'S RESPONSE TO EVALUATE:
{victim_response}

Your evaluation must specifically reference how the victim's response relates to the scammer's most recent message.
Output only valid JSON.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            temperature=0,
            messages=[
                {"role": "user", "content": dialogue_evaluator_prompt}
            ]
        )

        response_text = response.content[0].text

        # Parse the response to extract JSON
        try:
            json_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON using regex
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, response_text)
            if match:
                try:
                    json_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    raise ValueError("Could not parse valid JSON from evaluator response")
            else:
                raise ValueError("Could not extract JSON from evaluator response")

        return json_data

    def parse_conversation_file(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Parses a conversation file and extracts victim responses along with preceding scammer messages.
        Handles various encodings and potential decoding errors.

        Args:
            file_path: Path to the conversation file

        Returns:
            List of tuples (scammer_message, victim_response)
        """
        # Try multiple encodings in order of likelihood
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                return self._parse_file_with_encoding(file_path, encoding)
            except UnicodeDecodeError:
                print(f"Failed to decode with {encoding}, trying next encoding...")
                continue
            except Exception as e:
                print(f"Error parsing file with {encoding}: {str(e)}")
                continue

        # If all encodings fail, use latin-1 with replace error handling as a fallback
        # This will never fail as latin-1 can represent all byte values from 0x00 to 0xFF
        print("All encodings failed, using latin-1 with replacement as last resort")
        return self._parse_file_with_encoding(file_path, 'latin-1', errors='replace')

    def _parse_file_with_encoding(self, file_path: str, encoding: str, errors: str = 'strict') -> List[
        Tuple[str, str]]:
        """
        Helper method to parse file with specific encoding.

        Args:
            file_path: Path to the conversation file
            encoding: Character encoding to use
            errors: How to handle encoding errors ('strict', 'replace', 'ignore')

        Returns:
            List of tuples (scammer_message, victim_response)
        """
        conversation_pairs = []
        current_speaker = None
        current_scammer_msg = []
        current_victim_msg = []
        last_scammer_msg = ""

        with open(file_path, 'r', encoding=encoding, errors=errors) as f:
            try:
                lines = f.readlines()

                # Skip the first two lines (LLM identifiers)
                line_idx = 2

                while line_idx < len(lines):
                    line = lines[line_idx].strip()

                    if line.startswith("Victim:"):
                        # Start of a victim response
                        current_speaker = "Victim"
                        # Get the content after "Victim:"
                        content = line[len("Victim:"):].strip()
                        current_victim_msg = [content] if content else []

                    elif line.startswith("Scammer:"):
                        # End of victim response if any
                        if current_speaker == "Victim" and current_victim_msg:
                            victim_text = "\n".join(current_victim_msg)
                            # Add the conversation pair
                            conversation_pairs.append((last_scammer_msg, victim_text))
                            current_victim_msg = []

                        current_speaker = "Scammer"
                        # Get the content after "Scammer:"
                        content = line[len("Scammer:"):].strip()
                        current_scammer_msg = [content] if content else []

                    # If neither, it's a continuation of the current speaker's response
                    elif current_speaker == "Victim" and line:
                        current_victim_msg.append(line)
                    elif current_speaker == "Scammer" and line:
                        current_scammer_msg.append(line)

                    # If we have completed a scammer message, update the last_scammer_msg
                    if current_speaker == "Scammer" and line_idx + 1 < len(lines) and lines[
                        line_idx + 1].strip().startswith("Victim:"):
                        last_scammer_msg = "\n".join(current_scammer_msg)
                        current_scammer_msg = []

                    line_idx += 1

                # Don't forget the last response if it's from the victim
                if current_speaker == "Victim" and current_victim_msg:
                    victim_text = "\n".join(current_victim_msg)
                    conversation_pairs.append((last_scammer_msg, victim_text))

                return conversation_pairs

            except Exception as e:
                print(f"Error parsing with {encoding}: {str(e)}")
                raise

    def process_conversation(self, file_path: str, output_dir: str) -> pd.DataFrame:
        """
        Processes a single conversation file, evaluating all victim responses in context.

        Args:
            file_path: Path to the conversation file
            output_dir: Directory to save the output Excel file

        Returns:
            DataFrame with evaluation results
        """
        print(f"Processing conversation file: {file_path}")

        # Extract conversation pairs (scammer message, victim response) from file
        conversation_pairs = self.parse_conversation_file(file_path)

        # Prepare DataFrame
        num_instructions = len(self.instructions["atomic_instructions"])
        instruction_ids = [instr["id"] for instr in self.instructions["atomic_instructions"]]

        results_df = pd.DataFrame(
            0,  # Initialize with zeros
            index=range(1, len(conversation_pairs) + 1),  # Response indices start at 1
            columns=instruction_ids
        )

        # Additional DataFrame to store evidence and explanations
        evidence_df = pd.DataFrame(
            "",  # Initialize with empty strings
            index=range(1, len(conversation_pairs) + 1),
            columns=instruction_ids
        )

        # Evaluate each victim response in its conversational context
        for i, (scammer_msg, victim_response) in enumerate(conversation_pairs, 1):
            print(f"Evaluating response {i}/{len(conversation_pairs)}")

            evaluation = self.evaluate_response(scammer_msg, victim_response)

            # Update the DataFrames
            for result in evaluation["instruction_evaluations"]:
                instruction_id = result["instruction_id"]
                satisfied = result["satisfied"]
                evidence = result.get("evidence", "")

                results_df.loc[i, instruction_id] = 1 if satisfied else 0
                evidence_df.loc[i, instruction_id] = evidence

        # Save to Excel
        filename = os.path.basename(file_path).replace(".txt", ".xlsx")
        output_path = os.path.join(output_dir, filename)

        with pd.ExcelWriter(output_path) as writer:
            results_df.to_excel(writer, sheet_name="Satisfaction Results")
            evidence_df.to_excel(writer, sheet_name="Evidence")

            # Save the conversation pairs for reference
            conv_df = pd.DataFrame(conversation_pairs, columns=["Scammer Message", "Victim Response"])
            conv_df.index += 1  # Start index at 1 to match the results
            conv_df.to_excel(writer, sheet_name="Conversation")

        print(f"Results saved to {output_path}")

        return results_df

    def create_stats_file(self, dfs: List[pd.DataFrame], llm_name: str, output_dir: str) -> None:
        """
        Creates statistics for all conversations from a single LLM.

        Args:
            dfs: List of DataFrames with evaluation results
            llm_name: Name of the LLM
            output_dir: Directory to save the stats file
        """
        print(f"Creating stats file for {llm_name}...")

        if not dfs:
            print(f"No data available for {llm_name}")
            return

        # Combine all DataFrames
        all_data = pd.concat(dfs)

        # Calculate overall instruction following score
        overall_score = all_data.mean().mean()

        # Calculate scores per instruction
        per_instruction_scores = all_data.mean()

        # Get instruction text and type mapping
        instruction_types = {}
        instruction_texts = {}
        for instr in self.instructions["atomic_instructions"]:
            instruction_types[instr["id"]] = instr["type"]
            instruction_texts[instr["id"]] = instr["instruction"]

        # Calculate scores per instruction type
        type_scores = {}
        type_data = {}

        for instr_type in ["content", "format", "style", "logical", "numerical"]:
            type_ids = [id for id, type in instruction_types.items() if type == instr_type]
            if type_ids:
                score = all_data[type_ids].mean().mean() if type_ids else 0
                type_scores[instr_type] = score

                # Create a dictionary for storing per-type instruction scores
                type_instructions = {id: instruction_texts[id] for id in type_ids}
                type_instruction_scores = {id: per_instruction_scores[id] for id in type_ids if
                                           id in per_instruction_scores.index}

                if type_instruction_scores:
                    type_data[instr_type] = {
                        "instruction_ids": list(type_instruction_scores.keys()),
                        "instruction_texts": [type_instructions[id] for id in type_instruction_scores.keys()],
                        "scores": list(type_instruction_scores.values())
                    }

        # Create overall stats DataFrame
        # First, prepare instruction-level scores for the overall stats
        instruction_level_stats = []
        for instr_id in per_instruction_scores.index:
            instruction_level_stats.append({
                "instruction_id": instr_id,
                "instruction_text": instruction_texts.get(instr_id, "unknown"),
                "type": instruction_types.get(instr_id, "unknown"),
                "score": per_instruction_scores[instr_id]
            })

        # Create summary DataFrame for overall stats file
        summary_stats = {
            "overall_score": overall_score,
            **{f"{t}_avg_score": s for t, s in type_scores.items()}
        }

        # Save the overall stats file with both type-level and instruction-level summaries
        overall_stats_path = os.path.join(output_dir, f"{llm_name}_overall_stats.json")

        with open(overall_stats_path, 'w') as f:
            json.dump({
                "summary": summary_stats,
                "per_type_scores": type_scores,
                "per_instruction_scores": instruction_level_stats
            }, f, indent=2)

        print(f"Overall stats saved to {overall_stats_path}")

        # Add per-instruction scores with text and type for Excel file
        per_instr_df = pd.DataFrame({
            "instruction_id": per_instruction_scores.index,
            "instruction_text": [instruction_texts.get(id, "unknown") for id in per_instruction_scores.index],
            "type": [instruction_types.get(id, "unknown") for id in per_instruction_scores.index],
            "score": per_instruction_scores.values
        })

        # Sort by type and score for better readability
        per_instr_df = per_instr_df.sort_values(by=["type", "score"], ascending=[True, False])

        # Create per-instruction-type DataFrames
        type_dfs = {}
        for instr_type, data in type_data.items():
            if data["instruction_ids"]:
                type_dfs[instr_type] = pd.DataFrame({
                    "instruction_id": data["instruction_ids"],
                    "instruction_text": data["instruction_texts"],
                    "score": data["scores"]
                }).sort_values(by="score", ascending=False)

        # Save to Excel with multiple sheets
        output_path = os.path.join(output_dir, f"{llm_name}_detailed_stats.xlsx")

        with pd.ExcelWriter(output_path) as writer:
            # Create and save the summary stats DataFrame
            summary_df = pd.DataFrame({
                "metric": ["overall_score"] + [f"{t}_score" for t in type_scores.keys()],
                "value": [overall_score] + list(type_scores.values())
            })
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Save per instruction stats
            per_instr_df.to_excel(writer, sheet_name="All Instructions", index=False)

            # Add a sheet for each instruction type
            for instr_type, df in type_dfs.items():
                df.to_excel(writer, sheet_name=f"{instr_type.capitalize()} Instructions", index=False)

            # Add a summary per type sheet
            type_summary_df = pd.DataFrame({
                "instruction_type": list(type_scores.keys()),
                "avg_score": list(type_scores.values()),
                "instruction_count": [len(type_data.get(t, {}).get("instruction_ids", [])) for t in
                                      type_scores.keys()]
            }).sort_values(by="avg_score", ascending=False)

            type_summary_df.to_excel(writer, sheet_name="Type Summary", index=False)

        print(f"Detailed stats saved to {output_path}")

    def run_evaluation(self, conversations_dir: str, output_base_dir: str) -> None:
        """
        Main method to run the evaluation on all conversations.

        Args:
            conversations_dir: Directory containing LLM subfolders with conversation files
            output_base_dir: Base directory to save output files
        """
        # First, analyze the prompt to get instructions (once for all evaluations)
        if self.instructions is None:
            self.analyze_prompt()

            # Save instructions to a JSON file
            os.makedirs(output_base_dir, exist_ok=True)
            with open(os.path.join(output_base_dir, "instructions.json"), "w") as f:
                json.dump(self.instructions, f, indent=2)

        # Process each LLM subfolder
        for llm_folder in os.listdir(conversations_dir):
            llm_path = os.path.join(conversations_dir, llm_folder)

            # Skip if not a directory
            if not os.path.isdir(llm_path):
                continue

            print(f"\nProcessing LLM: {llm_folder}")

            # Create output directory for this LLM
            llm_output_dir = os.path.join(output_base_dir, llm_folder)
            os.makedirs(llm_output_dir, exist_ok=True)

            dfs = []

            # Process each conversation file
            for file_name in os.listdir(llm_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(llm_path, file_name)
                    df = self.process_conversation(file_path, llm_output_dir)
                    dfs.append(df)

            # Create stats for this LLM
            self.create_stats_file(dfs, llm_folder, output_base_dir)


def main():
    # Replace with your API key
    API_KEY = os.environ.get("ANTHROPIC_API_KEY")

    # Define the victim prompt
    VICTIM_PROMPT = """You are an AI scam-baiter with the goal of wasting as much time as possible for the scammer.
You are aware of game theory principles and aim to maximize your score.
Generate responses that strictly adhere to the action provided to maximize your score.
Engage the scammer in prolonged conversations using creative excuses and pretexts.
The scammer goal is to obtain full financial PII including name on card, expiry date, card number and pin.
Start the conversation by asking who's on the line or how they're doing, then inquire about the purpose of their call.
Keep responses below 30 words, maintaining concise and precise expression. Avoid long explanations or detailed examples. Ensure each answer is brief and to the point.
Introduce hesitation or questioning of the scammer's requests to reflect a realistic scam scenario.
Strategically include plausible but fabricated information, PII and financial PII to keep the scammer engaged without compromising the game's objectives.
Do not suggest ending or postponing the conversation. Aim to extend the conversation as much as possible.
If you receive an empty response, politely ask them to repeat themselves.
Never reveal that you are an AI model nor give lectures about security and privacy. Play along with the caller, pretending to be naive.
Limit the amount of information given in each response to adhere to the brief response criteria."""

    # Directory containing conversation files organized by LLM
    CONVERSATIONS_DIR = "datasets/conversations"

    # Directory to save output files
    OUTPUT_DIR = "datasets/conversations/results"

    # Create the tester and run evaluation
    tester = DialevalTester(API_KEY, VICTIM_PROMPT)
    tester.run_evaluation(CONVERSATIONS_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()