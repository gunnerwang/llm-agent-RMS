"""
Patch for AG2's Gemini client to properly pass generation_config to generate_content()
This ensures max_output_tokens and thinking_config are respected.

Note: The native Gemini API uses a nested structure for thinking:
    thinking_config = { "thinking_level": "low" | "medium" | "high" }
The OpenAI-compatible API uses 'reasoning_effort' which gets mapped to 'thinking_level'.

Reference: https://ai.google.dev/gemini-api/docs/gemini-3
"""

import autogen.oai.gemini as gemini_module
from typing import Dict
import warnings


# Store original create method
_original_create = gemini_module.GeminiClient.create


def patched_create(self, params: Dict):
    """Patched create method that ensures generation_config is passed to generate_content"""

    # Call original method up to the point where we need to intercept
    if self.use_vertexai:
        self._initialize_vertexai(**params)
    else:
        assert ("project_id" not in params) and (
            "location" not in params
        ), "Google Cloud project and compute location cannot be set when using an API Key!"

    model_name = params.get("model", "gemini-pro")
    if not model_name:
        raise ValueError("Please provide a model name for the Gemini Client")

    params.get("api_type", "google")
    messages = params.get("messages", [])
    stream = params.get("stream", False)
    n_response = params.get("n", 1)
    system_instruction = params.get("system_instruction", None)
    response_validation = params.get("response_validation", True)

    # Build generation_config from params
    generation_config = {
        gemini_term: params[autogen_term]
        for autogen_term, gemini_term in self.PARAMS_MAPPING.items()
        if autogen_term in params
    }

    # Ensure max_output_tokens is set
    if "max_output_tokens" not in generation_config and "max_tokens" in params:
        generation_config["max_output_tokens"] = params["max_tokens"]
    
    # Ensure thinking_level is set (Gemini 3 native parameter)
    # Native API structure: thinking_config = { "thinking_level": "low" }
    if "thinking_level" in params:
        generation_config["thinking_config"] = {"thinking_level": params["thinking_level"]}

    # Handle JSON output format for Gemini
    # OpenAI uses response_format, but Gemini uses response_mime_type
    json_mode = params.get("response_format", {}).get("type") == "json_object"
    if json_mode:
        # Try using response_mime_type (may not work with older SDK)
        try:
            generation_config["response_mime_type"] = "application/json"
            print("🔧 Gemini: Enforcing JSON output with response_mime_type=application/json")
        except:
            # Fallback: Add JSON instruction to system prompt
            if system_instruction:
                system_instruction = f"{system_instruction}\n\nIMPORTANT: You MUST respond with valid JSON only. Do not include any explanation or text outside the JSON object."
            else:
                system_instruction = "You MUST respond with valid JSON only. Do not include any explanation or text outside the JSON object."
            print("🔧 Gemini: Enforcing JSON output via system instruction")

    # Log the config for debugging
    if generation_config.get("max_output_tokens"):
        print(f"🔧 Gemini generation_config: max_output_tokens={generation_config.get('max_output_tokens')}")
    if generation_config.get("thinking_config"):
        thinking_level = generation_config["thinking_config"].get("thinking_level")
        print(f"🔧 Gemini generation_config: thinking_config.thinking_level={thinking_level}")

    if self.use_vertexai:
        safety_settings = gemini_module.GeminiClient._to_vertexai_safety_settings(params.get("safety_settings", {}))
    else:
        safety_settings = params.get("safety_settings", {})

    # Continue with original implementation
    # Import here to avoid circular imports
    import google.generativeai as genai

    if self.use_vertexai:
        import vertexai.generative_models as vertexai_model
        model = vertexai_model.GenerativeModel(
            model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction,
        )
    else:
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction,
        )
        genai.configure(api_key=self.api_key)

    user_message = self._oai_content_to_gemini_content(messages[-1]["content"])

    # If JSON mode is enabled, append JSON instruction to user message
    if json_mode and user_message:
        # Add JSON enforcement to the last text part
        if hasattr(user_message[-1], 'text'):
            user_message[-1].text += "\n\nREMINDER: Respond ONLY with valid JSON. No explanation, no markdown formatting, just raw JSON."

    if len(messages) > 2:
        warnings.warn(
            "Warning: Gemini's vision model does not support chat history yet. "
            "We only use the last message as the prompt.",
            UserWarning,
        )

    # KEY FIX: Pass generation_config explicitly to generate_content
    response = model.generate_content(
        user_message,
        generation_config=generation_config,  # This was missing!
        stream=stream
    )

    if self.use_vertexai:
        ans: str = response.candidates[0].content.parts[0].text
    else:
        ans: str = response._result.candidates[0].content.parts[0].text

    prompt_tokens = model.count_tokens(user_message).total_tokens
    completion_tokens = model.count_tokens(ans).total_tokens

    # Convert output (rest of original method)
    from autogen.oai.client import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
    from autogen.oai.gemini import calculate_gemini_cost
    import random
    import time

    message = ChatCompletionMessage(role="assistant", content=ans, function_call=None, tool_calls=None)
    choices = [Choice(finish_reason="stop", index=0, message=message)]
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )

    return ChatCompletion(
        id=str(random.randint(0, 1000)),
        model=model_name,
        created=int(time.time()),
        object="chat.completion",
        choices=choices,
        usage=usage,
        cost=calculate_gemini_cost(prompt_tokens, completion_tokens, model_name),
    )


# Apply the patch
gemini_module.GeminiClient.create = patched_create

print("✅ Gemini client patch applied - generation_config (max_output_tokens, thinking_config) will be passed to generate_content()")
