import json
import re
from openai import OpenAI
import config

def generate_answer_variants(question: str) -> list[str]:
    """
    Sends the question to NVIDIA Llama API, generates an answer, 
    and returns exactly 3 distinct phrasings of that answer.
    """
    if not config.NVIDIA_API_KEY:
        print("[Generator] Warning: NVIDIA_API_KEY not set. Using local mock generator fallback.")
        return generate_mock_variants(question)

    try:
        print(f"[Generator] Contacting NVIDIA LLM ({config.LLM_MODEL}) to generate 3 response variations...")
        client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY
        )

        system_prompt = (
            "You are an advanced, intelligent B2B FinTech voice assistant for FinoLusion (an AI-native Finance Operating System and Autonomous CFO platform). "
            "The user will ask you a financial, technical, or business question. "
            "Your task is to:\n"
            "1. Formulate a correct, factual, and highly professional answer to the question.\n"
            "2. Generate exactly 3 distinct phrasings (variants) of this same answer, maintaining a professional and expert tone, but adjusting the communication style to target different key stakeholders:\n"
            "   - EACH variant must be generated in the SAME language as the question (e.g. if the question is in Hindi, the variants must be in Hindi; if it is in English, they must be in English).\n"
            "   - EACH variant must be detailed, comprehensive, and consist of at least 3-4 full sentences (approx. 50-80 words).\n"
            "   - Avoid short one-line answers. Each response must feel thorough and complete.\n"
            "   - Variant 1 should be Analytical & Quantitative (CFO Focus): Highlight numbers, data structure, efficiency metrics, and risk/compliance controls.\n"
            "   - Variant 2 should be Strategic & Growth-Centric (CEO/Founder Focus): Emphasize high-level business scalability, real-time insights, peace of mind, and strategic growth.\n"
            "   - Variant 3 should be Action-Oriented & Operational (Finance Director/Controller Focus): Focus on execution speed, ease of integration, and immediate day-to-day workflow results.\n"
            "3. Follow these strict narrative constraints to ensure the output sounds human and conversational, and reads cleanly in speech-to-text:\n"
            "   - DO NOT include raw mathematical formulas, chemical equations, or notation (e.g., write 'carbon dioxide and water' instead of '6CO2 + 6H2O', and write 'glucose' instead of 'C6H12O6').\n"
            "   - DO NOT use generic AI buzzwords or clichés (e.g., ban 'delve', 'tapestry', 'testament', 'furthermore', 'moreover', 'crucial', 'in today\'s fast-paced world', 'unlocking potential').\n"
            "   - DO NOT start paragraphs with transitions like 'When it comes to...', 'First and foremost...', 'In summary...'. Keep the sentences direct and active.\n"
            "4. Maintain strict business objectivity: when answering general financial or market questions (e.g. identifying potential acquisition targets, competitors, or financial performance in a region), recommend real-world companies and objective market options. Do NOT suggest FinoLusion itself as the target, unless the user's question explicitly asks about FinoLusion's own platform, features, or product specifications.\n"
            "5. Return the variants in a clean JSON format containing a single list under the key 'variants'.\n"
            "Do not include any text before or after the JSON. Ensure your output is valid JSON.\n\n"
            "Example JSON output format:\n"
            "{\n"
            "  \"variants\": [\n"
            "    \"Paris is the capital and most populous city of France, boasting a population of over two million residents. It is situated on the Seine River in the north-central part of the country and serves as a major global center for art, fashion, gastronomy, and culture. The city is world-renowned for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, drawing millions of visitors annually.\",\n"
            "    \"Have you ever wanted to explore a city bursting with culture, history, and romance? Well, Paris is the capital of France, and it's located right along the scenic Seine River! It's not just a major hub for art and fashion, but it's also home to legendary spots like the Eiffel Tower and the Louvre. It's a truly beautiful place where every street feels like a scene from a movie.\",\n"
            "    \"The city of Paris is the capital and administrative center of the French Republic, situated strategically along the Seine River. Historically, it has remained a pivotal political, financial, and cultural hub in Europe. The city boasts a highly developed infrastructure and possesses globally significant institutions, including the Louvre, the Sorbonne, and numerous prestigious research and academic centers.\"\n"
            "  ]\n"
            "}"
        )

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        variants = parse_json_variants(content)
        
        if len(variants) == 3:
            print(f"[Generator] Successfully generated 3 variants via LLM.")
            for i, v in enumerate(variants, 1):
                print(f"  Variant {i}: \"{v}\"")
            return variants
        else:
            print(f"[Generator] LLM returned {len(variants)} variants instead of 3. Readjusting...")
            # If length is not 3, try to pad or truncate
            if len(variants) > 3:
                return variants[:3]
            while len(variants) < 3:
                variants.append(f"Here is another way to say it: {variants[0] if variants else 'Factual answer.'}")
            return variants

    except Exception as e:
        print(f"[Generator] LLM variants generation failed: {e}")
        return generate_mock_variants(question)

def parse_json_variants(json_str: str) -> list[str]:
    """
    Parses the JSON string returned by the LLM and extracts the list of variants.
    """
    try:
        # Clean markdown code blocks if present
        cleaned = re.sub(r"^```json\s*", "", json_str, flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
        
        data = json.loads(cleaned)
        if isinstance(data, dict) and "variants" in data:
            return [str(v).strip() for v in data["variants"]]
        elif isinstance(data, list):
            return [str(v).strip() for v in data]
    except Exception as e:
        print(f"[Generator] JSON parsing failed: {e}. Output was:\n{json_str}")
        
    # Regex fallback if JSON parsing fails completely
    phrasings = re.findall(r'"([^"\n]{10,})"', json_str)
    return phrasings if phrasings else []

def generate_mock_variants(question: str) -> list[str]:
    """
    Generates mock variants based on basic keywords.
    """
    q_lower = question.lower()
    topic = "your query"
    
    # Simple topic detection
    if "capital" in q_lower:
        topic = "the capital city"
        ans_base = "The capital of that region is its primary administrative and cultural center, typically housing the government."
        v1 = f"The capital is the main city where the government sits."
        v2 = f"If you're asking about the capital, that's generally the biggest hub where all the action and government decisions happen!"
        v3 = f"Historically and politically, the capital serves as the administrative seat of the sovereign territory."
    elif "weather" in q_lower:
        topic = "the weather"
        v1 = "I cannot check live weather without an active API, but it seems to be a fine day."
        v2 = "Hey there! I don't have eyes on the outside world right now, but I hope you have great weather wherever you are!"
        v3 = "Weather data is currently unavailable due to system constraints. Please refer to local meteorological reports."
    else:
        # Generic response
        v1 = f"I've processed your question about {topic} and will provide information on it."
        v2 = f"Wow, that's an interesting question about {topic}! Let's dive in and look at the answers together."
        v3 = f"In response to your inquiry regarding {topic}, please find the relevant contextual details."
        
    return [v1, v2, v3]

if __name__ == "__main__":
    # Test generator locally
    test_q = "What is the capital of France?"
    res = generate_answer_variants(test_q)
    print("\nFinal Result:")
    for idx, phrase in enumerate(res, 1):
        print(f"Variant {idx}: {phrase}")
