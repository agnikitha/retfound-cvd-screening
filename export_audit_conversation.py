import os
import json
import html
from datetime import datetime

source_log = r"C:\Users\Nikitha AG\.gemini\antigravity\brain\b66dc714-87bf-459d-a999-cd533aa75fa1\.system_generated\logs\transcript_full.jsonl"
out_dir = r"D:\new pro\conversation_audit_logs"
os.makedirs(out_dir, exist_ok=True)

md_out_path = os.path.join(out_dir, "complete_conversation_audit_transcript.md")
html_out_path = os.path.join(out_dir, "complete_conversation_audit_transcript.html")
jsonl_copy_path = os.path.join(out_dir, "transcript_full.jsonl")

# Copy raw jsonl
print("Copying raw transcript_full.jsonl...")
with open(source_log, 'rb') as src, open(jsonl_copy_path, 'wb') as dst:
    dst.write(src.read())

print("Parsing steps...")
steps = []
with open(source_log, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            steps.append(json.loads(line))

print(f"Total steps loaded: {len(steps)}")

# Group steps into conversation dialogue blocks
# A dialogue block begins with a USER_INPUT, followed by intermediate model/system steps, and model text content.
dialogue_blocks = []
current_block = None

for step in steps:
    stype = step.get('type')
    source = step.get('source')
    created_at = step.get('created_at', '')
    content = step.get('content', '')
    tool_calls = step.get('tool_calls', [])
    
    if stype == 'USER_INPUT' and source == 'USER_EXPLICIT':
        if current_block:
            dialogue_blocks.append(current_block)
        current_block = {
            'timestamp': created_at,
            'user_prompt': content,
            'assistant_responses': [],
            'actions': []
        }
    elif current_block is not None:
        if content and (source == 'MODEL' or stype == 'GENERIC'):
            # Model output to user
            # Avoid duplicate identical responses
            if not current_block['assistant_responses'] or current_block['assistant_responses'][-1] != content:
                current_block['assistant_responses'].append(content)
        if tool_calls:
            for tc in tool_calls:
                call_info = tc.get('tool_call', {})
                func_name = call_info.get('name', 'action')
                args = call_info.get('args', {})
                summary = args.get('toolSummary', '') or args.get('toolAction', '') or func_name
                current_block['actions'].append({
                    'tool': func_name,
                    'summary': summary
                })

if current_block:
    dialogue_blocks.append(current_block)

print(f"Total conversation turns identified: {len(dialogue_blocks)}")

# 1. Generate Markdown File
print("Generating Markdown file...")
with open(md_out_path, 'w', encoding='utf-8') as md:
    md.write("# COMPLETE AUDIT TRAIL: CONVERSATION & RESEARCH LOG\n\n")
    md.write(f"**Project:** RETFound-CVD & Ophthalmic Multi-Disease AI Screening  \n")
    md.write(f"**Conversation ID:** `b66dc714-87bf-459d-a999-cd533aa75fa1`  \n")
    md.write(f"**Audit Export Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    md.write(f"**Total Conversation Turns:** {len(dialogue_blocks)}  \n")
    md.write(f"**Total Audit Steps in Raw Transcript:** {len(steps)}  \n\n")
    md.write("---\n\n")
    md.write("## Table of Contents (User Requests)\n\n")
    for idx, blk in enumerate(dialogue_blocks, 1):
        clean_title = blk['user_prompt'].split('\n')[0][:80].replace('[', '').replace(']', '')
        md.write(f"- [Turn {idx}: {clean_title}](#turn-{idx})\n")
    md.write("\n---\n\n")

    for idx, blk in enumerate(dialogue_blocks, 1):
        md.write(f"## Turn {idx}\n")
        md.write(f"**Timestamp:** `{blk['timestamp']}`\n\n")
        md.write("### 👤 User Request:\n")
        md.write(f"> {blk['user_prompt']}\n\n")
        
        if blk['actions']:
            md.write("<details><summary>🛠️ Tools & Actions Executed in this Turn (" + str(len(blk['actions'])) + " actions)</summary>\n\n")
            for act in blk['actions'][:15]: # show up to 15 key actions
                md.write(f"- **{act['tool']}**: {act['summary']}\n")
            if len(blk['actions']) > 15:
                md.write(f"- *(...and {len(blk['actions']) - 15} more sub-steps)*\n")
            md.write("\n</details>\n\n")

        md.write("### 🤖 Assistant Response & Audit Deliverables:\n\n")
        if blk['assistant_responses']:
            for resp in blk['assistant_responses']:
                md.write(resp.strip() + "\n\n")
        else:
            md.write("*(Autonomous background processing and tool execution)*\n\n")
        md.write("---\n\n")

# 2. Generate Interactive HTML File
print("Generating Interactive HTML file...")
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audit Trail: Complete Conversation Log - RETFound CVD</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #24292e;
    background-color: #f6f8fa;
    margin: 0;
    padding: 0;
  }}
  .header {{
    background: linear-gradient(135deg, #1f2937, #111827);
    color: #ffffff;
    padding: 30px 40px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }}
  .header h1 {{ margin: 0 0 10px 0; font-size: 26px; font-weight: 700; }}
  .header p {{ margin: 4px 0; color: #9ca3af; font-size: 14px; }}
  .badge {{
    display: inline-block;
    padding: 4px 10px;
    background-color: #10b981;
    color: white;
    font-size: 12px;
    font-weight: 600;
    border-radius: 9999px;
    margin-top: 10px;
  }}
  .container {{
    max-width: 1100px;
    margin: 30px auto;
    padding: 0 20px;
  }}
  .controls {{
    background: white;
    padding: 15px 25px;
    border-radius: 8px;
    margin-bottom: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .search-box {{
    padding: 8px 12px;
    width: 350px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
  }}
  .turn-card {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    border: 1px solid #e5e7eb;
    overflow: hidden;
  }}
  .turn-header {{
    background: #f9fafb;
    padding: 12px 20px;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    color: #374151;
    font-size: 14px;
  }}
  .user-box {{
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 15px 20px;
    margin: 15px 20px;
    border-radius: 0 6px 6px 0;
  }}
  .user-box h4 {{ margin: 0 0 6px 0; color: #1d4ed8; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .user-box pre {{ margin: 0; font-family: inherit; font-size: 15px; white-space: pre-wrap; font-weight: 500; color: #1e3a8a; }}
  .assistant-box {{
    padding: 10px 25px 20px 25px;
  }}
  .assistant-box h4 {{ margin: 15px 0 10px 0; color: #047857; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .content {{
    font-size: 15px;
    color: #1f2937;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .tools-box {{
    background: #fdf2f8;
    border-left: 4px solid #ec4899;
    padding: 10px 15px;
    margin: 10px 20px;
    border-radius: 4px;
    font-size: 13px;
    color: #831843;
  }}
  .tools-box summary {{ font-weight: 600; cursor: pointer; }}
  .tools-list {{ margin: 8px 0 0 15px; padding: 0; }}
  .tools-list li {{ margin-bottom: 4px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #d1d5db;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{ background-color: #f3f4f6; }}
</style>
</head>
<body>

<div class="header">
  <h1>Complete Conversation & Audit Trail Transcript</h1>
  <p><strong>Project:</strong> RETFound-CVD Non-Invasive Atherosclerosis & Multi-Disease Ophthalmic AI</p>
  <p><strong>Conversation ID:</strong> b66dc714-87bf-459d-a999-cd533aa75fa1</p>
  <p><strong>Export Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <strong>Total Turns:</strong> {len(dialogue_blocks)} | <strong>Total Actions:</strong> {len(steps)}</p>
  <span class="badge">Official Clinical & Research Audit Log</span>
</div>

<div class="container">
  <div class="controls">
    <div><strong>Viewing:</strong> All Recorded Dialogue from Inception to Present</div>
    <input type="text" id="searchInput" class="search-box" placeholder="Search conversations, keywords, or metrics..." onkeyup="filterConversations()">
  </div>
"""

for idx, blk in enumerate(dialogue_blocks, 1):
    u_text = html.escape(blk['user_prompt'])
    html_content += f"""
  <div class="turn-card" id="turn-{idx}">
    <div class="turn-header">
      <span>Turn #{idx}</span>
      <span>{html.escape(blk['timestamp'])}</span>
    </div>
    <div class="user-box">
      <h4>👤 User Prompt</h4>
      <pre>{u_text}</pre>
    </div>
"""
    if blk['actions']:
        html_content += f"""
    <div class="tools-box">
      <details>
        <summary>🛠️ Tools & Actions Executed ({len(blk['actions'])} actions recorded)</summary>
        <ul class="tools-list">
"""
        for act in blk['actions'][:12]:
            html_content += f"<li><strong>{html.escape(act['tool'])}:</strong> {html.escape(str(act['summary']))}</li>\n"
        if len(blk['actions']) > 12:
            html_content += f"<li><em>...and {len(blk['actions']) - 12} additional actions</em></li>\n"
        html_content += """
        </ul>
      </details>
    </div>
"""

    html_content += """
    <div class="assistant-box">
      <h4>🤖 Assistant Audit Response</h4>
"""
    if blk['assistant_responses']:
        for resp in blk['assistant_responses']:
            # escape html
            r_text = html.escape(resp.strip())
            html_content += f"<div class=\"content\">{r_text}</div>\n"
    else:
        html_content += "<p><em>(Autonomous execution and data processing completed)</em></p>\n"

    html_content += """
    </div>
  </div>
"""

html_content += """
</div>

<script>
function filterConversations() {
  var input = document.getElementById('searchInput');
  var filter = input.value.toLowerCase();
  var cards = document.getElementsByClassName('turn-card');
  for (var i = 0; i < cards.length; i++) {
    var text = cards[i].innerText.toLowerCase();
    if (text.indexOf(filter) > -1) {
      cards[i].style.display = "";
    } else {
      cards[i].style.display = "none";
    }
  }
}
</script>
</body>
</html>
"""

with open(html_out_path, 'w', encoding='utf-8') as hf:
    hf.write(html_content)

print("Export complete!")
print(f"MD: {md_out_path}")
print(f"HTML: {html_out_path}")
print(f"JSONL: {jsonl_copy_path}")
