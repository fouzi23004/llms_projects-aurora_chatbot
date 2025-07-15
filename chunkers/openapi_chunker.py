

def chunk_openapi_spec(spec: dict, base_url: str, page_title: str):
    chunks = []
    paths = spec.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            section_title = f"{method.upper()} {path}"
            lines = [f"### [{method.upper()}] {path}"]

            if 'summary' in details:
                lines.append(f"**Summary:** {details['summary']}")
            if 'description' in details:
                lines.append(f"**Description:** {details['description']}")
            if 'operationId' in details:
                lines.append(f"**Operation ID:** `{details['operationId']}`")
            if 'tags' in details:
                lines.append(f"**Tags:** {', '.join(details['tags'])}")
            if 'consumes' in details:
                lines.append(f"**Consumes:** {', '.join(details['consumes'])}")
            if 'produces' in details:
                lines.append(f"**Produces:** {', '.join(details['produces'])}")

            # Parameters
            parameters = details.get("parameters", [])
            if parameters:
                lines.append("**Parameters:**")
                for param in parameters:
                    location = param.get("in", "unknown")
                    name = param.get("name", "unnamed")
                    desc = param.get("description", "")
                    required = param.get("required", False)
                    lines.append(f"- `{name}` ({location}){' (required)' if required else ''}: {desc}")

            # Responses
            responses = details.get("responses", {})
            if responses:
                lines.append("**Responses:**")
                for status_code, resp in responses.items():
                    if "$ref" in resp:
                        ref = resp["$ref"]
                        lines.append(f"- `{status_code}`: See `{ref}`")
                    else:
                        desc = resp.get("description", "No description")
                        lines.append(f"- `{status_code}`: {desc}")

            chunks.append({
                "url": base_url,
                "page_title": page_title,
                "section_title": section_title,
                "content": "\n".join(lines)
            })

    return chunks

