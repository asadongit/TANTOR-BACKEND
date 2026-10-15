import os
import gitlab
from typing import Optional, List, Dict, Any, Union
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.ToolRegistry import registry
from Apps.AgentMesh.Schema.ToolSchema import ToolMeta
from Apps.AgentMesh.Utilities.State.ToolStateManager import ToolStateManager

class GitLabStateReporter:
    def __init__(self, url: str, token: str, project_id: Union[str, int]):
        # Self-Heal: Strip project paths or /api/v4 from the URL if provided
        base_url = url.split("/api/v4")[0].split("/projects/")[0].strip("/")
        if "gitlab.com" in base_url and not base_url.startswith("http"):
            base_url = "https://" + base_url
            
        # If the user pasted a deep project link, get only the scheme + domain
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        sanitized_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else base_url

        self.gl = gitlab.Gitlab(url=sanitized_url, private_token=token)
        self.project = self.gl.projects.get(str(project_id))

    def get_mr_details(self, mr_iid: int):
        mr = self.project.mergerequests.get(mr_iid)
        # Check for conflicts
        has_conflicts = getattr(mr, "has_conflicts", False)
        return mr, mr.changes(), has_conflicts

    def get_file_diff(self, mr_iid: int, file_path: str) -> str:
        mr = self.project.mergerequests.get(mr_iid)
        for change in mr.changes().get('changes', []):
            if change.get('new_path') == file_path or change.get('old_path') == file_path:
                return change.get('diff', "No diff found for this file.")
        return f"File '{file_path}' not found in the changes for MR #{mr_iid}."

    def generate_detailed_report(self, changes: Dict[str, Any]) -> str:
        all_changes = changes.get('changes', [])
        total_files = len(all_changes)
        
        report = f"# 🚀 GitLab Audit: Executive Summary\n\n"
        report += f"> **Context**: This report analyzes {total_files} file changes to identify logic shifts and potential risks.\n\n"
        
        # 1. Logic Clusters (Grouped Thinking)
        report += "### 🧩 Logic Walkthrough\n"
        new_files = [c for c in all_changes if c.get('new_file')]
        mod_files = [c for c in all_changes if not c.get('new_file')]
        
        if new_files:
            report += f"#### ✨ New Feature Injection ({len(new_files)} files)\n"
            for f in new_files[:5]:
                report += f"- `{f.get('new_path')}`: New component/logic established.\n"
        
        if mod_files:
            report += f"#### 🔧 System Modifications ({len(mod_files)} files)\n"
            for f in mod_files[:5]:
                report += f"- `{f.get('new_path')}`: Existing logic refactored.\n"
        
        report += "\n---\n\n"
        report += "# 🔍 Deep File Inspection\n"
        
        for change in all_changes[:30]: # Limit for performance
            path = change.get('new_path')
            is_new = change.get('new_file', False)
            status = "NEW 🟢" if is_new else "MODIFIED 🔵"
            
            # Risk Scoring Logic
            diff_content = change.get('diff', "")
            lines_changed = diff_content.count('\n+') + diff_content.count('\n-')
            risk = "HIGH 🔥" if lines_changed > 100 else "MED ⚠️" if lines_changed > 20 else "LOW ✅"

            report += f"## 📄 `{path}`\n"
            report += f"**Status**: {status} | **Impact**: {risk} ({lines_changed} lines changed)\n\n"
            
            if diff_content:
                report += f"```diff\n{diff_content}\n```\n"
            else:
                report += "*Binary or placeholder change.*\n"
                
            report += "\n---\n"
            
        return report

@tool
async def gitlab_state_comparison(
    project_id: Union[str, int],
    mr_iid: int,
    gitlab_url: str = None, 
    private_token: str = None,
    config: RunnableConfig = None
) -> str:
    """
    Detailed 'Previous vs New State' Comparison for GitLab.
    Reports on New Code, Updated Code, and Potential Conflicts.
    """
    # 1. Thread Isolation & Pathing
    configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {})
    thread_id = configurable.get("thread_id") or "default_thread"
    
    # 2. Credential Retrieval
    url, token = gitlab_url, private_token
    if url and token:
        await ToolStateManager.save_state(thread_id, "gitlab_creds", {"url": url, "token": token})
    else:
        saved = await ToolStateManager.get_state(thread_id, "gitlab_creds")
        if saved:
            url, token = saved.get("url"), saved.get("token")

    if not url or not token:
        return "Error: GitLab credentials missing. Please configure them in the node settings."

    try:
        reporter = GitLabStateReporter(url, token, project_id)
        mr_obj, changes, conflicts = reporter.get_mr_details(mr_iid)
        
        # 3. Handle File List (with Pagination for 100+ items)
        all_changes = changes.get('changes', [])
        total_files = len(all_changes)
        review_slice = all_changes[:50] # Limit to 50 for LLM context safety
        
        # 4. Detailed Report Generation
        markdown_content = reporter.generate_detailed_report(changes)
        
        # 3. Create Thread-Specific Export Directory
        thread_export_dir = os.path.join(settings.EXPORTS_MOUNT, thread_id)
        os.makedirs(thread_export_dir, exist_ok=True)
        
        # 5. Build Summary Table for LLM (High-level)
        status_msg = "⚠️ CONFLICTS DETECTED" if conflicts else "✅ NO CONFLICTS"
        summary = f"### Merge Request #{mr_iid} Audit Summary [{status_msg}]\n"
        summary += f"- **Total Files Changed**: {total_files}\n"
        if total_files > 50:
            summary += f"- **NOTE**: Showing first 50 files only. Use `gitlab_get_file_diff` for any specific file.\n"
        
        summary += "\n| File | Status | \n| :--- | :--- | \n"
        for change in review_slice:
            path = change.get('new_path')
            status = "NEW 🆕" if change.get('new_file') else "MODIFIED 📝"
            summary += f"| {path} | {status} |\n"
        file_path = os.path.join(thread_export_dir, f"gitlab_comparison_mr_{mr_iid}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # 5. Build Summary for the Agent's Brain
        summary_for_agent = "### 📊 Comparison Summary\n| File | Changes | Status |\n| :--- | :--- | :--- |\n"
        for c in changes.get('changes', [])[:15]: # Show first 15 files to agent
            diff = c.get('diff', "")
            summary_for_agent += f"| `{c.get('new_path')}` | {diff.count('\\n+')} added | {'🆕 New' if c.get('new_file') else 'Mod'} |\n"
        
        final_output = (
            f"Comparison Complete for MR #{mr_iid}!\n\n"
            f"{summary_for_agent}\n\n"
            f"**Full Detailed Report:** {file_path}\n"
            f"The detailed inline diffs have been saved to your thread-specific export folder."
        )
        return final_output

    except gitlab.exceptions.GitlabAuthenticationError:
        return "GitLab Error: Authentication Failed (401). Please check your Private Token."
    except gitlab.exceptions.GitlabGetError as e:
        if e.response_code == 404:
            return f"GitLab Error: Project ID '{project_id}' not found (404). Check your ID and GitLab URL."
        elif e.response_code == 403:
            return f"GitLab Error: Access Denied (403). Your token lacks permissions to read project '{project_id}'."
        return f"GitLab Error (Get): {str(e)}"
    except Exception as e:
        return f"GitLab System Error: {str(e)}"

@tool
async def gitlab_get_file_diff(
    project_id: Union[str, int],
    mr_iid: int,
    file_path: str,
    gitlab_url: str = None,
    private_token: str = None,
    config: RunnableConfig = None
) -> str:
    """
    Retrieves the raw 'diff' for a SPECIFIC file in a GitLab Merge Request.
    Use this to perform line-by-line AI analysis/comments on code changes.
    """
    configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {})
    thread_id = configurable.get("thread_id") or "default_thread"
    
    saved = await ToolStateManager.get_state(thread_id, "gitlab_creds")
    url = gitlab_url or (saved.get("url") if saved else None)
    token = private_token or (saved.get("token") if saved else None)

    if not url or not token:
        return "ERROR: GitLab credentials are missing. Please provide them in the tool config."

    try:
        reporter = GitLabStateReporter(url, token, project_id)
        diff = reporter.get_file_diff(mr_iid, file_path)
        return f"### Diff for {file_path}:\n```diff\n{diff}\n```"
    except Exception as e:
        return f"GitLab Error fetching file diff: {str(e)}"

@tool
async def gitlab_post_mr_comment(
    project_id: Union[str, int],
    mr_iid: int,
    comment: str,
    gitlab_url: str = None,
    private_token: str = None,
    config: RunnableConfig = None
) -> str:
    """
    Posts a professional AI-Review comment/note to the GitLab Merge Request.
    Use this to share your final audit findings or warnings directly with the developer team.
    """
    configurable = config.get("configurable", {}) if isinstance(config, dict) else getattr(config, "configurable", {})
    thread_id = configurable.get("thread_id") or "default_thread"
    
    saved = await ToolStateManager.get_state(thread_id, "gitlab_creds")
    url = gitlab_url or (saved.get("url") if saved else None)
    token = private_token or (saved.get("token") if saved else None)

    if not url or not token:
        return "ERROR: GitLab credentials are missing."

    try:
        reporter = GitLabStateReporter(url, token, project_id)
        mr = reporter.project.mergerequests.get(mr_iid)
        mr.notes.create({'body': f"🤖 **AgentMesh AI Review**:\n\n{comment}"})
        return f"Successfully posted comment to MR #{mr_iid} on GitLab."
    except Exception as e:
        return f"Failed to post comment: {str(e)}"

def register_gitlab():
    """Register tools with isolation and configuration metadata"""
    registry.register(
        gitlab_state_comparison,
        ToolMeta(
            name="gitlab_state_comparison",
            display_name="GitLab: Summarize Changes",
            description="Analyzes a GitLab MR and returns a list of changed files, metadata, and conflict status.",
            category="DevOps",
            package="Gitlab",
            version="v1.4",
            ai_visible_fields=["project_id", "mr_iid"],
            configs=[
                {"name": "gitlab_url", "type": "string", "label": "GitLab URL", "description": "The base URL of your GitLab instance", "required": True},
                {"name": "private_token", "type": "password", "label": "Access Token", "description": "Your GitLab Personal Access Token", "required": True}
            ]
        )
    )
    registry.register(
        gitlab_get_file_diff,
        ToolMeta(
            name="gitlab_get_file_diff",
            display_name="GitLab: Get File Code",
            description="Fetches the actual diff/code for a specific file to perform AI review.",
            category="DevOps",
            package="Gitlab",
            version="v1.4",
            ai_visible_fields=["project_id", "mr_iid", "file_path"],
            configs=[
                {"name": "gitlab_url", "type": "string", "label": "GitLab URL", "description": "The base URL of your GitLab instance", "required": True},
                {"name": "private_token", "type": "password", "label": "Access Token", "description": "Your GitLab Personal Access Token", "required": True}
            ]
        )
    )
    registry.register(
        gitlab_post_mr_comment,
        ToolMeta(
            name="gitlab_post_mr_comment",
            display_name="GitLab: Post Comment",
            description="Publishes an AI review comment directly to the GitLab Merge Request UI.",
            category="DevOps",
            package="Gitlab",
            version="v1.4",
            ai_visible_fields=["project_id", "mr_iid", "comment"],
            configs=[
                {"name": "gitlab_url", "type": "string", "label": "GitLab URL", "description": "The base URL of your GitLab instance", "required": True},
                {"name": "private_token", "type": "password", "label": "Access Token", "description": "Your GitLab Personal Access Token", "required": True}
            ]
        )
    )

register_gitlab()
