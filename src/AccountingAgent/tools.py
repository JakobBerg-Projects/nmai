"""Tool definitions exposed to the LLM via function-calling.

Defined in OpenAI format; the LLM abstraction converts to Anthropic format
when needed.
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "tripletex_request",
            "description": (
                "Make an HTTP request to the Tripletex v2 REST API. "
                "Use this tool for every interaction with Tripletex: "
                "creating, reading, updating, and deleting resources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "description": "HTTP method.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "API path relative to the base URL, e.g. '/employee', "
                            "'/customer/42', '/invoice'. Always starts with '/'."
                        ),
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Query parameters as key-value pairs. "
                            "Use for GET filters like fields, count, from, name, etc."
                        ),
                        "additionalProperties": True,
                    },
                    "json_body": {
                        "type": "object",
                        "description": (
                            "JSON request body for POST and PUT requests. "
                            "Omit for GET and DELETE."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["method", "path"],
            },
        },
    },
]
