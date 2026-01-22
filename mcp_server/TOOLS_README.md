# Adding New Tools to MCP Server

The MCP server now uses dynamic tool discovery from the `/tools/` directory. Each tool is organized in its own subfolder.

## Directory Structure

```
/tools/
  /get_sum/
    schema.json
    handler.py
  /get_product/
    schema.json
    handler.py
  /calculate_average/
    schema.json
    handler.py
  /your_new_tool/          ← Add new tools here
    schema.json
    handler.py
```

## Creating a New Tool

### Step 1: Create Tool Folder
Create a new subfolder in `/tools/` with a descriptive name (e.g., `/string_concatenate/`)

### Step 2: Create `schema.json`
Define the tool's metadata and input schema:

```json
{
  "name": "string-concatenate",
  "description": "Concatenates two strings together",
  "inputSchema": {
    "type": "object",
    "properties": {
      "str1": {
        "type": "string",
        "description": "First string"
      },
      "str2": {
        "type": "string",
        "description": "Second string"
      }
    },
    "required": ["str1", "str2"],
    "additionalProperties": false
  }
}
```

**JSON Schema Types Supported:**
- `string` → Python `str`
- `number` → Python `float`
- `integer` → Python `int`
- `boolean` → Python `bool`
- `array` → Python `list`
- `object` → Python `dict`

### Step 3: Create `handler.py`
Implement the tool's logic with a function named `handler`:

```python
"""Handler for string-concatenate tool"""


def handler(str1: str, str2: str) -> str:
    """Concatenates two strings together"""
    result = str1 + str2
    return f"Concatenated result: {result}"


# For async handlers, use async def:
# async def handler(param1: str) -> str:
#     """Async tool example"""
#     return f"Result: {param1}"
```

**Key Requirements:**
- Function must be named `handler`
- Parameter names must match the schema's property names
- Supports both sync and async functions
- Return value should be a string
- Function receives arguments as keyword arguments (`**kwargs`)

## Examples

### Example 1: Text Length Tool

**schema.json:**
```json
{
  "name": "get-text-length",
  "description": "Returns the length of a text string",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The text to measure"
      }
    },
    "required": ["text"],
    "additionalProperties": false
  }
}
```

**handler.py:**
```python
"""Handler for get-text-length tool"""


def handler(text: str) -> str:
    """Get the length of text"""
    length = len(text)
    return f"The text '{text}' has {length} characters."
```

### Example 2: Async Tool - File Reader

**schema.json:**
```json
{
  "name": "read-file",
  "description": "Reads content from a file (relative to project root)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "filename": {
        "type": "string",
        "description": "Relative path to file to read"
      }
    },
    "required": ["filename"],
    "additionalProperties": false
  }
}
```

**handler.py:**
```python
"""Handler for read-file tool"""
import asyncio
from pathlib import Path


async def handler(filename: str) -> str:
    """Read a file asynchronously"""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            return f"Error: File not found - {filename}"
        
        content = file_path.read_text()
        return f"File content:\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

### Example 3: Array Input - List Statistics

**schema.json:**
```json
{
  "name": "list-stats",
  "description": "Returns min, max, and average of a list of numbers",
  "inputSchema": {
    "type": "object",
    "properties": {
      "values": {
        "type": "array",
        "items": {
          "type": "number"
        },
        "description": "List of numbers"
      }
    },
    "required": ["values"],
    "additionalProperties": false
  }
}
```

**handler.py:**
```python
"""Handler for list-stats tool"""


def handler(values: list) -> str:
    """Calculate statistics for a list of numbers"""
    if not values:
        return "Error: Empty list provided"
    
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)
    
    return f"Min: {min_val}, Max: {max_val}, Average: {avg_val:.2f}"
```

## Testing Your Tools

Once you add a new tool:

1. **Restart the MCP server** - Tools are loaded on startup
2. **Check logs** - Look for `✓ Loaded tool: <tool-name>` messages
3. **Test in chat** - Ask the AI to use your new tool

Example: "Add 10 and 20" → Uses `get-sum` tool
         "What is the average of 5, 10, 15?" → Uses `calculate-average` tool
         "Concatenate hello and world" → Uses `string-concatenate` tool (if added)

## Troubleshooting

### Tool not appearing
- Check tool folder is in `/tools/` directory
- Verify `schema.json` exists and is valid JSON
- Verify `handler.py` exists and contains a `handler` function
- Check MCP server logs for errors

### Tool errors when called
- Check parameter names in `handler.py` match schema properties
- Ensure handler returns a string
- Check function signature matches the schema (correct types)
- Look at MCP server logs for exception details

### Handler function signature mismatch
```python
# ✓ Correct - matches schema properties
def handler(a: float, b: float) -> str:
    pass

# ✗ Wrong - parameter names don't match
def handler(x: float, y: float) -> str:
    pass
```

## Tool Auto-Discovery

The server automatically discovers and loads tools on startup:
1. Scans `/tools/` directory
2. For each subfolder, loads `schema.json` and `handler.py`
3. Dynamically imports handler functions
4. Registers all tools with the MCP server
5. No need to edit code - just add folders!
