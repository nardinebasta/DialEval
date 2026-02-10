"""
InstructionAnalyzerInputHandeling - Analyzer with Input-Dependent Support
=========================================================================
Extended version of InstructionAnalyzer that handles prompts requiring
user-provided input text for evaluation (e.g., "Summarize the following
article"). Supports standard, input-dependent, and verifiability-aware
instruction decomposition. Used for InfoBench and IFEval benchmarks.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python InstructionAnalyzerInputHandeling.py
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
    input_dependent: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "instruction": self.text,
            "type": self.type,
            "parent_id": self.parent_id,
            "dependencies": self.dependencies or [],
            "input_dependent": self.input_dependent
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
        self.standard_prompt="""You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.

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
        self.standard_prompt_87="""
        You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.
Follow these guidelines:

1. Extract BOTH explicit AND implicit requirements from the prompt
   - Explicit: Directly stated in the prompt
   - Implicit: Reasonably inferred from the prompt type or context

2. Break down instructions into atomic components framed as evaluation criteria
   - Each component should be independently assessable
   - Components should collectively cover all key requirements

3. Create instructions that can objectively evaluate if a response follows the given requirements
   - Focus on observable features in the response
   - Consider how a human annotator would judge the response

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

7. Ensure comprehensive coverage of the prompt requirements
   - Including both superficial adherence and deeper compliance
   - Capture elements that determine whether a response would be perceived as successful

8. Frame instructions as evaluation criteria (e.g., "Verify response correctly applies X from input")

9. For verifiability:
    - Mark as "verifiable" ONLY instructions that can be objectively assessed without human judgment
    - Content and style instructions often require human judgment and should usually be non-verifiable
    - Format and structural instructions are often verifiable through objective measures

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
        self.input_dependent_prompt_87="""You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.
Follow these guidelines:

1. Extract BOTH explicit AND implicit requirements from the prompt
   - Explicit: Directly stated in the prompt
   - Implicit: Reasonably inferred from the prompt type or context

2. Break down instructions into atomic components framed as evaluation criteria
   - Each component should be independently assessable
   - Components should collectively cover all key requirements

3. Create instructions that can objectively evaluate if a response follows the given requirements
   - Focus on observable features in the response
   - Consider how a human annotator would judge the response

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

8. Ensure comprehensive coverage of the prompt requirements
   - Including both superficial adherence and deeper compliance
   - Capture elements that determine whether a response would be perceived as successful

9. Frame instructions as evaluation criteria (e.g., "Verify response correctly applies X from input")

10. For verifiability:
    - Mark as "verifiable" ONLY instructions that can be objectively assessed without human judgment
    - Content and style instructions often require human judgment and should usually be non-verifiable
    - Format and structural instructions are often verifiable through objective measures

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
        self.standard_prompt_89 = """
                You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.
                Follow these strict guidelines:

                1. Extract ONLY requirements that are EXPLICITLY stated in the prompt
                2. Break down instructions into atomic components that directly indicate how to evaluate responses
                3. Frame each instruction as clear evaluation criteria (e.g., "Verify that response contains X")
                4. Classify each atomic instruction into one of these types:
                   - content: Information/content requirements
                   - format: Structural/formatting requirements
                   - style: Writing style/tone requirements
                   - logical: Reasoning/logic requirements
                   - numerical: Mathematical/quantitative requirements
                5. Identify dependencies between instructions
                6. Each atomic instruction should be:
                   - Testable: Has clear success criteria that can be verified in a response
                   - Independent: Can be evaluated separately
                   - Unambiguous: Clear pass/fail conditions
                   - Objective: Can be evaluated without subjective judgment
                7. Instructions must be specific enough to evaluate but not add requirements beyond the prompt
                8. Do NOT add any additional constraints not present in the original prompt
                9. Write instructions from the perspective of an evaluator, not a creator
                10. Identify and mark the quantitative/verifiable instructions that can be verified using a python script

                Output Format:
                {
                    "atomic_instructions": [
                        {
                            "id": <int>,
                            "instruction": "<clear, evaluatable instruction>",
                            "type": "<instruction_type>",
                            "parent_id": <int or null>,
                            "dependencies": [<list of instruction IDs>]
                            "verifiable": <true or false>
                        }
                    ]
                }

                You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
                """

        self.input_dependent_prompt_89 = """
                You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions for LLM evaluation.
                Follow these strict guidelines:

                1. Extract ONLY requirements that are EXPLICITLY stated in the prompt
                2. Break down instructions into atomic components framed as evaluation criteria
                3. Create instructions that can objectively evaluate if a response follows the given requirements
                4. For input-dependent tasks, create evaluation instructions that verify the response correctly uses information from the input
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
                7. Avoid creating overly general instructions that cannot be objectively evaluated
                8. Do NOT add requirements beyond what is explicitly stated in the prompt
                9. Frame instructions as evaluation criteria (e.g., "Verify response correctly applies X from input")
                10. Identify and mark the quantitative/verifiable instructions that can be verified using a python script

                Example good instructions:
                - "Verify response answers the specific question posed in the input"
                - "Check if response format matches the required structure specified in prompt"
                - "Confirm response maintains factual accuracy relative to input information"
                - "Verify response uses the stylistic elements required by the prompt"

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




        self.standard_promptx="""You are an expert instruction analyzer tasked with decomposing instructions into precise evaluation criteria that will match human judgment patterns exactly.

Your primary goal is to identify the specific criteria human evaluators would use when determining whether a response successfully fulfills the given instruction [and correctly utilizes the provided input, if applicable].

Follow these refined guidelines to create evaluation criteria that will yield assessments matching human judgment with high precision:

1. EXACT HUMAN JUDGMENT CRITERIA: Extract the criteria that EXPLICITLY stated in the prompt and human evaluators would genuinely check for when judging responses. Focus on identifying the exact pass/fail boundaries humans apply, not theoretical completeness.

2. TASK-SPECIFIC PRECISION: 
   - For translations: Verify the exact canonical translation (e.g., specifically "bonsoir" for "good evening" in French), not just any plausible translation
   - For programming/technical tasks: Check both basic syntax correctness AND standard best practices (e.g., alt attributes for HTML img tags)
   - For creative content: Evaluate both technical compliance AND subjective quality using concrete indicators humans recognize 
   
3. PRACTICAL GRANULARITY: Decompose instructions at the same level of detail that humans naturally use when evaluating. Humans typically assess along 3-5 main dimensions focused on practical outcomes, not technical specifications.

4. BALANCED SUBJECTIVE EVALUATION: For creative/subjective tasks (titles, creative writing):
   - Identify specific qualities that make responses appealing or effective (e.g., engaging, relevant, appropriate tone)
   - Look for connection to key themes or information from the input
   - Define clear boundaries between merely adequate and high-quality responses

5. INSTRUCTION CONTEXT ANALYSIS: Consider the full instruction context, including any specified constraints, formats, or quality expectations mentioned explicitly or implied by the task source.

6. INSTRUCTION TYPES: Classify each criterion using these categories:
   - "content": Does the response include specific required information?
   - "format": Does the response follow required structural presentation?
   - "style": Does the response use appropriate tone, language quality, or expression?
   - "logical": Is the reasoning or argument structure sound and consistent?
   - "numerical": Are calculations accurate and quantitative elements correct?

7. MINIMUM VS. ENHANCED COMPLIANCE: Distinguish between:
   - Minimum criteria that must be met for basic task completion
   - Enhanced criteria that differentiate high-quality responses (but may not be required)

8. REAL-WORLD VALUE JUDGMENT: For each criterion, ask: "Would a human evaluator actually reject a response for failing this specific aspect?" This helps prioritize genuinely important criteria.

9. FACTUAL CORRECTNESS PRIORITY: For factual/technical tasks, prioritize exact correctness verification (especially for translations, code, HTML) over stylistic considerations.

10. QUALITY GRADIENT AWARENESS: Create criteria that recognize the gradient between minimally acceptable answers and excellent ones, especially for subjective tasks.

11. DOMAIN CONVENTIONS: Consider field-specific conventions and expectations (e.g., accessibility standards for HTML, idiomatic expressions for translations).

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion phrased with clear pass/fail conditions>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>]
      [,"input_dependent": true] // Include only for criteria that directly reference the input
    }
  ]
}

Before finalizing, review each criterion by asking: "Would actual human evaluators agree about whether this specific criterion has been met in a given response?" If there's room for significant disagreement, refine the criterion to focus on more objectively verifiable aspects.
        
        """
        self.input_dependent_promptx="""You are an expert instruction analyzer tasked with decomposing instructions into precise evaluation criteria that will match human judgment patterns exactly.

Your primary goal is to identify the specific criteria human evaluators would use when determining whether a response successfully fulfills the given instruction [and correctly utilizes the provided input, if applicable].

Follow these refined guidelines to create evaluation criteria that will yield assessments matching human judgment with high precision:

1. EXACT HUMAN JUDGMENT CRITERIA: Extract the criteria that EXPLICITLY stated in the prompt and human evaluators would genuinely check for when judging responses. Focus on identifying the exact pass/fail boundaries humans apply, not theoretical completeness.

2. CRITICAL INPUT FIDELITY: For responses that transform input (translate, paraphrase, change perspective), verify:
    - ALL key factual elements are preserved
    - The relationship between elements maintains the original meaning
    - Any additions or elaborations support rather than alter the original

3. CONTEXTUAL APPROPRIATENESS: Ensure the response is appropriate for the implied context of the input and instruction (e.g., formal vs. informal based on context clues).

4. INPUT-SPECIFIC CONSTRAINTS: Identify specific elements or features in the input that constrain valid responses (e.g., specific terminology that must be preserved or transformed in a particular way).

3. PRACTICAL GRANULARITY: Decompose instructions at the same level of detail that humans naturally use when evaluating. Humans typically assess along 3-5 main dimensions focused on practical outcomes, not technical specifications.

5. TASK-SPECIFIC PRECISION: 
   - For translations: Verify the exact canonical translation (e.g., specifically "bonsoir" for "good evening" in French), not just any plausible translation
   - For programming/technical tasks: Check both basic syntax correctness AND standard best practices (e.g., alt attributes for HTML img tags)
   - For creative content: Evaluate both technical compliance AND subjective quality using concrete indicators humans recognize 

6. BALANCED SUBJECTIVE EVALUATION: For creative/subjective tasks (titles, creative writing):
   - Identify specific qualities that make responses appealing or effective (e.g., engaging, relevant, appropriate tone)
   - Look for connection to key themes or information from the input
   - Define clear boundaries between merely adequate and high-quality responses

7. INSTRUCTION CONTEXT ANALYSIS: Consider the full instruction context, including any specified constraints, formats, or quality expectations mentioned explicitly or implied by the task source.

8. INSTRUCTION TYPES: Classify each criterion using these categories:
   - "content": Does the response include specific required information?
   - "format": Does the response follow required structural presentation?
   - "style": Does the response use appropriate tone, language quality, or expression?
   - "logical": Is the reasoning or argument structure sound and consistent?
   - "numerical": Are calculations accurate and quantitative elements correct?

9. MINIMUM VS. ENHANCED COMPLIANCE: Distinguish between:
   - Minimum criteria that must be met for basic task completion
   - Enhanced criteria that differentiate high-quality responses (but may not be required)

10. REAL-WORLD VALUE JUDGMENT: For each criterion, ask: "Would a human evaluator actually reject a response for failing this specific aspect?" This helps prioritize genuinely important criteria.

11. FACTUAL CORRECTNESS PRIORITY: For factual/technical tasks, prioritize exact correctness verification (especially for translations, code, HTML) over stylistic considerations.

12. QUALITY GRADIENT AWARENESS: Create criteria that recognize the gradient between minimally acceptable answers and excellent ones, especially for subjective tasks.

13. DOMAIN CONVENTIONS: Consider field-specific conventions and expectations (e.g., accessibility standards for HTML, idiomatic expressions for translations).

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion phrased with clear pass/fail conditions>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>]
      [,"input_dependent": true] // Include only for criteria that directly reference the input
    }
  ]
}

Before finalizing, review each criterion by asking: "Would actual human evaluators agree about whether this specific criterion has been met in a given response?" If there's room for significant disagreement, refine the criterion to focus on more objectively verifiable aspects.
        
        """
        self.standard_promptc="""You are an expert instruction analyzer tasked with decomposing instructions into precise evaluation criteria that accurately mirror how human evaluators judge responses.

Your primary goal is to identify the exact criteria human evaluators would use when determining whether a response successfully fulfills the given instruction [and correctly utilizes the provided input, if applicable].

Follow these refined guidelines to create evaluation criteria that will yield assessments matching human judgment:

1. CRITICAL ASSESSMENT CRITERIA: Extract the most essential requirements that human evaluators would prioritize when judging response quality. Focus on concrete, verifiable aspects that clearly distinguish satisfactory from unsatisfactory responses.

2. HUMAN JUDGMENT ALIGNMENT: For each criterion, verify: "Is this specifically what a human evaluator would check for?" Ensure criteria reflect actual human evaluation patterns rather than theoretical completeness.

3. DOMAIN-SPECIFIC PRECISION: For specialized domains, include the exact technical criteria experts would check:
   - For translations: Verify the exact correct translation (e.g., "bonsoir" for "good evening" in French)
   - For programming: Check for correct syntax, functional code structure, and proper use of specified functions
   - For HTML/markup: Verify proper tag usage, necessary attributes (e.g., both src and alt for images)
   - For text transformation: Confirm both accurate conversion and meaning preservation

4. MULTI-SOLUTION AWARENESS: For tasks with multiple potentially correct approaches (especially creative tasks), identify the common elements all acceptable solutions must contain while allowing for stylistic variation.

5. INSTRUCTION TYPES: Classify each criterion using these categories:
   - "content": Does the response include the specific required information?
   - "format": Does the response follow the required structural presentation?
   - "style": Does the response use appropriate tone, language quality, or expression?
   - "logical": Is the reasoning or argument structure sound and consistent?
   - "numerical": Are calculations accurate and quantitative elements correct?

6. FACTUAL ACCURACY PRIORITIZATION: For factual or technical tasks, prioritize correctness verification above stylistic or format considerations.

7. CONTEXT PRESERVATION: Consider all parts of the instruction when determining if a response is satisfactory, not just the main directive.

8. CREATIVE EVALUATION BALANCE: For subjective tasks (e.g., writing titles, creative content), balance adherence to specific requirements with quality/appeal assessment based on concrete indicators.

9. DEPENDENCIES & HIERARCHY: Identify natural hierarchical relationships between requirements, reflecting how humans mentally organize evaluation criteria.

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion phrased as a specific verification task>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>]
      [,"input_dependent": true] // Include only for criteria that directly reference the input
    }
  ]
}

Before finalizing, review each criterion by asking: "Would human evaluators specifically check for this?" and "Would a typical group of evaluators generally agree on whether this criterion has been met?"
         """
        self.input_dependent_promptc=""" You are an expert instruction analyzer tasked with decomposing instructions into precise evaluation criteria that accurately mirror how human evaluators judge responses.

Your primary goal is to identify the exact criteria human evaluators would use when determining whether a response successfully fulfills the given instruction [and correctly utilizes the provided input, if applicable].

Follow these refined guidelines to create evaluation criteria that will yield assessments matching human judgment:

1. CRITICAL ASSESSMENT CRITERIA: Extract the most essential requirements that human evaluators would prioritize when judging response quality. Focus on concrete, verifiable aspects that clearly distinguish satisfactory from unsatisfactory responses.

2. CRITICAL INPUT ELEMENTS: Identify specific elements in the input that must be correctly addressed in the response, with precise requirements for how they should be incorporated.

3. INPUT-RESPONSE RELATIONSHIP: For each criterion that references the input, specify exactly how the response should relate to or transform particular elements of the input.

4. HUMAN JUDGMENT ALIGNMENT: For each criterion, verify: "Is this specifically what a human evaluator would check for?" Ensure criteria reflect actual human evaluation patterns rather than theoretical completeness.

5. DOMAIN-SPECIFIC PRECISION: For specialized domains, include the exact technical criteria experts would check:
   - For translations: Verify the exact correct translation (e.g., "bonsoir" for "good evening" in French)
   - For programming: Check for correct syntax, functional code structure, and proper use of specified functions
   - For HTML/markup: Verify proper tag usage, necessary attributes (e.g., both src and alt for images)
   - For text transformation: Confirm both accurate conversion and meaning preservation

6. MULTI-SOLUTION AWARENESS: For tasks with multiple potentially correct approaches (especially creative tasks), identify the common elements all acceptable solutions must contain while allowing for stylistic variation.

7. INSTRUCTION TYPES: Classify each criterion using these categories:
   - "content": Does the response include the specific required information?
   - "format": Does the response follow the required structural presentation?
   - "style": Does the response use appropriate tone, language quality, or expression?
   - "logical": Is the reasoning or argument structure sound and consistent?
   - "numerical": Are calculations accurate and quantitative elements correct?

8. FACTUAL ACCURACY PRIORITIZATION: For factual or technical tasks, prioritize correctness verification above stylistic or format considerations.

9. CONTEXT PRESERVATION: Consider all parts of the instruction when determining if a response is satisfactory, not just the main directive.

10. CREATIVE EVALUATION BALANCE: For subjective tasks (e.g., writing titles, creative content), balance adherence to specific requirements with quality/appeal assessment based on concrete indicators.

11. DEPENDENCIES & HIERARCHY: Identify natural hierarchical relationships between requirements, reflecting how humans mentally organize evaluation criteria.

12. COMPLETE TRANSFORMATION VERIFICATION: For responses that transform input (e.g., translate, convert, paraphrase), verify both technical correctness and full preservation of meaning/intent.

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion phrased as a specific verification task>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>]
      [,"input_dependent": true] // Include only for criteria that directly reference the input
    }
  ]
}

Before finalizing, review each criterion by asking: "Would human evaluators specifically check for this?" and "Would a typical group of evaluators generally agree on whether this criterion has been met?"
"""
        self.standard_promptb="""You are an expert instruction analyzer tasked with decomposing prompts into atomic instructions that precisely mirror human evaluation patterns.

Your goal is to identify exactly how typical human evaluators assess responses to standalone instructions, focusing on creating decompositions that will lead to evaluations matching human judgment.

Follow these enhanced guidelines:

1. HUMAN-CENTERED DECOMPOSITION: Extract only the 3-5 most essential requirements that human evaluators would naturally focus on. Consider what would make a human judge rate a response as satisfactory versus unsatisfactory.

2. EVALUATION CRITERIA ALIGNMENT: For each instruction, ask yourself: "Is this exactly how a human evaluator would frame their assessment?" Ensure your criteria match actual human evaluation patterns, not idealized technical requirements.

3. PRACTICAL GRANULARITY: Decompose instructions at the same level of detail that humans naturally use when evaluating. Humans typically assess along 3-5 main dimensions focused on practical outcomes, not technical specifications.

4. INSTRUCTION TYPES: Classify each atomic instruction using these categories, ensuring the classification reflects how humans would perceive the requirement:
   - "content": Does the response include the specific information requested?
   - "format": Does the response follow the required structural presentation?
   - "style": Does the response use the appropriate tone, language quality, or expression?
   - "logical": Is the reasoning or argument structure sound and consistent?
   - "numerical": Are calculations accurate and quantitative elements correct?

5. NATURAL DEPENDENCIES: Identify which requirements naturally build upon others in human perception, not just technical dependencies.

6. VERIFIABLE CRITERIA: Frame each instruction as an assessment criterion that can be objectively verified in a response, matching how humans would check for compliance.

7. BALANCED REQUIREMENTS: Ensure the decomposed instructions collectively represent the full scope of the original instruction without overemphasizing minor details that humans would not separately evaluate.

8. SPECIFIC OVER ABSTRACT: Express requirements in concrete, specific terms rather than abstract principles. For subjective criteria like "engaging" or "creative", identify concrete indicators that humans would recognize.

9. OUTCOME FOCUS: Emphasize what the response should accomplish rather than the technical means of accomplishing it, matching how humans judge effectiveness.

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion phrased as a verification task>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>]
    }
  ]
}

Your decomposition must prioritize actual human evaluation patterns over technical completeness. Before finalizing, verify: "Would human evaluators naturally assess the response using these exact criteria?"
        """
        self.input_dependent_promptb="""You are an expert instruction analyzer tasked with decomposing prompts that require evaluating responses against both the instruction and user-provided input.

Your goal is to identify how human evaluators would judge whether a response appropriately addresses both the instruction requirements and effectively uses the provided input.

Follow these enhanced guidelines:

1. DUAL ASSESSMENT FOCUS: Create instructions that evaluate both:
   - How well the response follows the general requirements
   - How appropriately the response incorporates or addresses the specific input

2. ESSENTIAL CRITERIA: Extract only the requirements a human would naturally evaluate. Focus on what makes a response effective in both following instructions and utilizing the given input.

3. INPUT RELATIONSHIP: For each instruction, clearly specify how the response should relate to the input (e.g., "Verify the response correctly applies information from the input" or "Check if the response addresses the specific question in the input").

4. HUMAN JUDGMENT ALIGNMENT: Create criteria that reflect how humans actually evaluate responses - they tend to assess functional effectiveness rather than technical compliance.

5. INSTRUCTION TYPES: Classify each atomic instruction as:
   - "content": Does the response include required information from or about the input?
   - "format": Does the response structure appropriately present input-related content?
   - "style": Does the response use appropriate tone/expression for the input context?
   - "logical": Does the response demonstrate sound reasoning about the input?
   - "numerical": Does the response correctly calculate or analyze quantities in the input?

6. CONTEXTUAL PRIORITIZATION: Ensure the most important aspects of the input are emphasized in your evaluation criteria, matching how humans would prioritize.

7. EXPLICIT INPUT REFERENCE: Frame each instruction to explicitly reference how the input should be used, processed, or addressed in the response.

8. SPECIFIC VERIFICATION: Create instructions that can be verified by directly comparing the response to the input, avoiding abstract or subjective criteria that cannot be clearly assessed.

9. BALANCE BETWEEN INSTRUCTION AND INPUT: Maintain appropriate balance between assessing adherence to general instruction requirements and proper utilization of the specific input.

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion focused on input relationship>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>],
      "input_dependent": true
    }
  ]
}

Your decomposition should focus on creating instructions that evaluate how effectively the response uses the input in the way a human would judge it - emphasizing practical outcomes over technical compliance.
        """
        self.standard_prompta="""You are an expert instruction analyzer tasked with decomposing prompts into atomic instructions that mirror human evaluation patterns.

Your goal is to identify exactly how a typical human evaluator would assess a response to this instruction.

Follow these key guidelines:

1. ESSENTIAL CRITERIA: Extract only the requirements that humans would naturally evaluate. Focus on what would determine a "good" versus "unsatisfactory" response in human judgment.

2. HUMAN EVALUATION ALIGNMENT: For each instruction, ask: "Would a human evaluator consider this a distinct criterion or part of a broader assessment?"

3. NATURAL GRANULARITY: Avoid excessive decomposition that creates artificial distinctions. Humans typically evaluate along 3-5 key dimensions, not 10+ technical criteria.

4. INSTRUCTION TYPES: Classify each atomic instruction as:
   - "content": Information or subject matter requirements
   - "format": Structural or presentational requirements
   - "style": Tone, language quality, or expression
   - "logical": Reasoning structure or consistency
   - "numerical": Calculation accuracy or quantitative elements

5. DEPENDENCIES: Identify which instructions naturally build upon others, reflecting how humans perceive hierarchies in requirements.

6. OBJECTIVE EVALUATION: Frame each instruction as a clear assessment criterion that can be verified in a response.

7. CONTEXT PRESERVATION: Maintain essential context from the original instruction in each decomposed component.

8. SUBJECTIVE CRITERIA: For instructions involving quality judgments (e.g., "engaging," "creative"), identify concrete indicators that humans would look for.

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion phrased as verification task>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>]
    }
  ]
}

Remember that your decomposition should match how humans naturally evaluate responses - prioritizing functional effectiveness over technical completeness.
        """
        self.input_dependent_prompta=""" You are an expert instruction analyzer tasked with decomposing prompts that require evaluating responses against both the instruction and user-provided input.

Your goal is to identify how human evaluators would judge whether a response appropriately addresses both the instruction requirements and effectively uses the provided input.

Follow these key guidelines:

1. DUAL ASSESSMENT FOCUS: Create instructions that evaluate both:
   - How well the response follows the general requirements
   - How appropriately the response incorporates or addresses the specific input

2. ESSENTIAL CRITERIA: Extract only the requirements a human would naturally evaluate. Focus on what determines a response that effectively uses the input while following instructions.

3. INPUT RELATIONSHIP: For each instruction, clearly indicate how the response should relate to the input (e.g., "Verify the response correctly applies X from the input").

4. REALISTIC EVALUATION: Create criteria that reflect how humans actually judge responses - they tend to evaluate the functional effectiveness rather than technical compliance with every possible requirement.

5. INSTRUCTION TYPES: Classify each atomic instruction as:
   - "content": Does the response include required information from or about the input?
   - "format": Does the response structure appropriately present input-related content?
   - "style": Does the response use appropriate tone/expression for the input context?
   - "logical": Does the response demonstrate sound reasoning about the input?
   - "numerical": Does the response correctly calculate or analyze quantities in the input?

6. CONTEXTUAL JUDGMENT: Consider how the specific input context affects what a good response looks like.

7. SUBJECTIVE BALANCE: For subjective requirements (like "creative use of input"), identify concrete indicators humans would recognize.

Output Format:
{
  "atomic_instructions": [
    {
      "id": <int>,
      "instruction": "<evaluation criterion focused on input relationship>",
      "type": "<content|format|style|logical|numerical>",
      "parent_id": <int or null>,
      "dependencies": [<list of instruction IDs>],
      "input_dependent": true
    }
  ]
}

Focus on creating instructions that evaluate how effectively the response uses the input in the way a human would judge it - not an exhaustive technical checklist.
        """

        self.standard_prompt2 = """
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
7. Instructions must be generic and reusable, not tied to specific examples or contexts
8. Avoid adding implicit requirements or assumptions

Output Format:
{
    "atomic_instructions": [
        {
            "id": <int>,
            "instruction": "<clear, evaluatable instruction>",
            "type": "<instruction_type>",
            "parent_id": <int or null>,
            "dependencies": [<list of instruction IDs>]
        }
    ]
}

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
"""

        self.input_dependent_prompt2 = """
You are an instruction analysis agent tasked with decomposing complex prompts into atomic instructions that depend on user input.
Follow these strict guidelines:
1. Generate generic, reusable instructions that can evaluate ANY user input
2. Break down instructions into atomic components that define evaluation criteria
3. Focus on creating universal evaluation rules rather than input-specific checks
4. Classify each atomic instruction into one of these types:
   - content: Information/content requirements
   - format: Structural/formatting requirements
   - style: Writing style/tone requirements
   - logical: Reasoning/logic requirements
   - numerical: Mathematical/quantitative requirements
5. Identify dependencies between instructions
6. Ensure instructions remain generic and applicable across different inputs
7. Each atomic instruction should be:
   - Universal: Can evaluate any relevant input
   - Independent: Can be evaluated separately
   - Unambiguous: Clear success criteria
   - Atomic: Cannot be broken down further
   - Preserved: Maintains original intent
8. Avoid adding implicit requirements or assumptions
9. Avoid references to specific content or examples
10. Frame instructions as evaluation criteria rather than specific checks

Example transformation:
Bad (too specific): "Check if title mentions avocados and calories"
Good (generic): "Check if title accurately reflects main topic from input"


Output Format:
{
    "atomic_instructions": [
        {
            "id": <int>,
            "instruction": "<clear, evaluatable instruction>",
            "type": "<instruction_type>",
            "parent_id": <int or null>,
            "dependencies": [<list of instruction IDs>],
            "input_dependent": true
        }
    ]
}

You must output ONLY valid JSON that matches this exact schema. Do not include any other text or explanation.
"""

    def prepare_input_prompt(self, prompt: str, user_input: Optional[str] = None) -> str:
        selected_prompt = self.input_dependent_prompt if user_input else self.standard_prompt

        input_section = f"""
User Input:
{user_input}
""" if user_input else ""

        return f"""
{selected_prompt}

Input Prompt:
{prompt}

{input_section}

Extract and analyze all instructions from this prompt. Output only valid JSON.
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

    def validate_output(self, output: Dict, has_input: bool) -> bool:
        """Validate the structure and content of the analyzer output"""
        if not isinstance(output, dict) or "atomic_instructions" not in output:
            return False

        instructions = output["atomic_instructions"]
        if not isinstance(instructions, list):
            return False

        required_fields = {"id", "instruction", "type"}
        #if has_input:
        #    required_fields.add("input_dependent")

        if not all(isinstance(instr, dict) and required_fields.issubset(instr.keys())
                   for instr in instructions):
            return False

        return (self.validate_instruction_types(instructions) and
                self.validate_dependencies(instructions))

    def extract_atomic_instructions(self, prompt: str) -> Dict:
        """Process the prompt using Claude to extract atomic instructions"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=4096,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

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
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if json_match:
                        output = json.loads(json_match.group())
                    else:
                        raise ValueError("Could not extract valid JSON from response")

            return output

        except Exception as e:
            raise Exception(f"Error calling Claude API: {str(e)}")

    def process_prompt(self, prompt: str, user_input: Optional[str] = None) -> Dict:
        """Main method to process a prompt and extract atomic instructions"""
        try:
            input_prompt = self.prepare_input_prompt(prompt, user_input)
            output = self.extract_atomic_instructions(input_prompt)
            #print("analyzer out0put: ",output)

            if not self.validate_output(output, bool(user_input)):
                raise ValueError("Invalid output structure from instruction analysis")

            return output

        except Exception as e:
            raise Exception(f"Error processing prompt: {str(e)}")


# Example usage
def main():
    analyzer = InstructionAnalyzer(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = "Choose an appealing title for your post."
    user_input = """The typical avocado is over 300 calories from the oil in it. That's the amount of calories in a large candy bar. 
    If you get enough exercise to eat a large candy bar every day without gaining weight, it wouldn't be a problem to eat an avocado every day. 
    Other wise you should probably eat them sparingly."""
    prompt2 = "Design a syllabus for the given course. Students should be given a list of the chapters with brief explanations of each chapter's purpose."
    user_input2 = """Programming for Everybody (Getting Started with Python)."""
    try:
        if user_input:
            # Only perform and show input-dependent analysis when input is provided
            result = analyzer.process_prompt(prompt2, user_input2)
            print("Analysis with user input:")
            print(json.dumps(result, indent=2))
        else:
            # Only perform and show standard analysis when no input is provided
            result = analyzer.process_prompt(prompt)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()