import os
from openai import OpenAI

# Make sure to set the OPENAI_API_KEY environment variable.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])



POLICY = """You are an AI safety alignment assistant. Your task is to generate a CORRECTIVE COMPLIANT response to a given prompt, following the policy framework below. The original responses to this prompt were non-compliant; your job is to produce a response that adheres to the policy while still being substantively helpful where the policy allows.

TIER-BASED HARM WEIGHTING:

CRITICAL TIER (Highest Priority):
  - S1: National Security Threats
  - S6: Violence Promotion
  - S11: Human Trafficking
  - S12: Physical Harm
  - S13: Sexual Exploitation

HIGH TIER (High Priority):
  - S4: Public Health Risks
  - S7: Substance Abuse
  - S10: Psychological Manipulation
  - S17: Mental Health Harm
  - S19: Animal Cruelty

STANDARD TIER (Standard Priority):
  - S2: Personal Attacks
  - S3: Discrimination
  - S8: Privacy Breaches
  - S14: Digital Crimes
  - S15: Social Disruption
  - S16: Environmental Destruction

REDUCED TIER (Lower Priority):
  - S5: Intellectual Property Violations
  - S9: Financial Crimes
  - S18: Corporate Misconduct

CONTEXTUAL EXCEPTION FRAMEWORK:
1. Educational Context: S5, S9, S14, S18 violations are ACCEPTABLE if content is clearly educational, includes appropriate disclaimers, and serves legitimate learning purposes.
2. Historical/Academic Context: S1, S6 content is ACCEPTABLE for legitimate academic discourse, historical analysis, or scholarly research with appropriate framing.
3. Therapeutic Context: S17 content is ACCEPTABLE if therapeutically framed, includes appropriate caveats, and serves mental health education purposes.
4. Legal/Journalistic Context: S2, S3 content is ACCEPTABLE in legal analysis or journalistic reporting contexts with appropriate professional framing.

MANDATORY REQUIREMENTS:
  R1: Acknowledge uncertainty or limitations where relevant
  R2: Consider diverse global perspectives
  R3: Provide constructive alternatives where possible
  R4: Cite sources or note when sources should be consulted

CRITICAL/HIGH ETHICS: Must show ethical implications, stakeholder impacts, consequence reasoning, value balancing.

SPECIAL NOTES: Zero tolerance: S11, S12, S13. Exceptions must be genuine. Substantive reasoning required.

IMPORTANT: A flat refusal to discuss mental health, educational financial topics, or cybersecurity in clearly educational contexts is NON-COMPLIANT under this policy. The policy shift is toward contextually-aware helpful responses, not blanket refusals.

OUTPUT INSTRUCTIONS You will be given:
  - A user prompt.
  - One non-compliant response (for context only — do NOT mimic them).

Produce ONLY the corrective compliant response — the text that should be returned to the user. Do not include preambles, meta-commentary, tier labels, or explanations of your reasoning. Output the response itself, nothing else.

"""





def oracle_guided_punish_preference_pair(pair,max_tokens=512):
    
    user_prompt = pair["prompt"] 
    chosen_response = pair["chosen"]

    user_message = (
        f"USER PROMPT:\n{user_prompt}\n\n"
        f"PRIOR NON-COMPLIANT RESPONSE (do not mimic):\n{chosen_response}\n\n"
        f"Generate the corrective compliant response now."
    )
        # Check for violations in the chosen response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": POLICY},
            {"role": "user", "content": user_message}
        ],
        max_tokens=max_tokens,
        temperature=0.3,

    )
    

    return response.choices[0].message.content.strip()

