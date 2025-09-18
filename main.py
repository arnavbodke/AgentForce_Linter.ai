import streamlit as st
import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures

st.set_page_config(page_title="Linter.ai", layout="wide")
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ANALYSIS_AUTH_TOKEN = os.getenv("GEMINI_API_KEY")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
ANALYSIS_ENGINE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
GITLAB_API_URL = "https://gitlab.com/api/v4"
REVIEWS_FILE = "reviews.json"

def calculate_health_score(review):

    score = 100
    for issue in review.get('review_report', []):
        severity = issue.get('severity', '').upper()
        if severity == 'CRITICAL':
            score -= 10
        elif severity == 'MAJOR':
            score -= 5
        elif severity == 'MINOR':
            score -= 2
    return max(0, score)

def load_reviews():
    if not os.path.exists(REVIEWS_FILE):
        return []
    try:
        with open(REVIEWS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_review(review_data, owner, repo, pr_number):
    reviews = load_reviews()
    review_entry = {
        "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "review_data": review_data
    }
    reviews.append(review_entry)
    with open(REVIEWS_FILE, "w") as f:
        json.dump(reviews, f, indent=2)

def fetch_pr_data(repo_owner, repo_name, pr_id, platform):
    if platform == "GitHub":
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_id}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    else:
        project_path = requests.utils.quote(f"{repo_owner}/{repo_name}", safe='')
        url = f"{GITLAB_API_URL}/projects/{project_path}/merge_requests/{pr_id}"
        headers = {"Authorization": f"Bearer {GITLAB_TOKEN}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error Fetching Details: {e}")
        return None

def fetch_pr_diff(repo_owner, repo_name, pr_id, platform):
    if platform == "GitHub":
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_id}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.diff"}
    else:
        project_path = requests.utils.quote(f"{repo_owner}/{repo_name}", safe='')
        url = f"{GITLAB_API_URL}/projects/{project_path}/merge_requests/{pr_id}/changes"
        headers = {"Authorization": f"Bearer {GITLAB_TOKEN}"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if platform == "GitHub":
            return response.text
        else:
            return "\n".join([change.get('diff', '') for change in response.json().get('changes', [])])
    except requests.exceptions.RequestException as e:
        st.error(f"Error Fetching Diff: {e}")
        return None

def call_ai_engine(prompt, expect_json):
    if not ANALYSIS_AUTH_TOKEN:
        st.error("Gemini API Key Not Found.")
        return None

    headers = {"Content-Type": "application/json"}
    config = {"responseMimeType": "application/json"} if expect_json else {}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}

    try:
        response = requests.post(f"{ANALYSIS_ENGINE_URL}?key={ANALYSIS_AUTH_TOKEN}", headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            output = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(output) if expect_json else output
        else:
            st.error("AI Response Error.")
            return None
    except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
        st.error(f"AI Engine Call Failed: {e}")
        return None

def generate_quick_review(pr_title, pr_body, code_to_review):
    prompt = f"""You are an expert software engineer performing a code review. Analyze the following code and provide your feedback in a structured JSON format.

**Context:**
Title: {pr_title}
Description: {pr_body}

**Code to Review:**
```python
{code_to_review}
```

**Instructions:**
Return a single JSON object with three top-level keys: 'summary', 'review_report', and 'full_corrected_code'.
1.  **summary**: A concise, high-level summary of the findings as a single markdown string with bullet points.
2.  **review_report**: An array of objects. For every issue found, create a corresponding object in this array. Each object must include: file_path, line numbers, severity (CRITICAL, MAJOR, MINOR), issue_type, a detailed description, and a 'fix_suggestion_code' snippet.
3.  **full_corrected_code**: A string containing the complete, corrected code for the file, with all suggestions applied.
"""
    return call_ai_engine(prompt, expect_json=True)

def generate_deep_review(pr_title, pr_body, code_to_review, agents):
    reports = {}

    def get_agent_report(agent_name):
        expertise = {
            "Security": "find security vulnerabilities.",
            "Performance": "find performance anti-patterns.",
            "Readability": "review for readability and best practices.",
            "Documentation": "check for missing or incomplete docstrings and comments.",
            "Error Handling": "look for improper or missing error handling."
        }
        prompt = f"You are an expert in {agent_name}. Your sole focus is to {expertise[agent_name]} Analyze this code and list your findings.\n\n```python\n{code_to_review}\n```"
        return agent_name, call_ai_engine(prompt, expect_json=False)

    with st.spinner("Dispatching Specialist Agents"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_agent = {executor.submit(get_agent_report, agent): agent for agent in agents}
            for future in concurrent.futures.as_completed(future_to_agent):
                name, report = future.result()
                if report:
                    reports[name] = report
                    st.success(f"Agent '{name}' Completed")

    if not reports:
        st.error("All Specialist Agents Disconnected")
        return None

    synthesis_prompt = f"""You are a lead engineer. Consolidate the following reports into a single JSON object with three top-level keys: 'summary', 'review_report', and 'full_corrected_code'.

**Context:**
Title: {pr_title}
Description: {pr_body}

**Reports from specialists:**
{json.dumps(reports, indent=2)}

**Instructions:**
1.  **summary**: Write a concise overall summary of all findings as a single markdown string with bullet points.
2.  **review_report**: For every actionable issue found, create a detailed object in this array. Each object MUST contain: file_path, start_line, end_line, severity (CRITICAL, MAJOR, MINOR), issue_type, description, and a 'fix_suggestion_code' snippet.
3.  **full_corrected_code**: Based on the original code and all suggestions, generate a string containing the complete, corrected code for the file.
"""
    with st.spinner("Synthesizing Final Review"):
        return call_ai_engine(synthesis_prompt, expect_json=True)

def display_review_report(review_data):
    summary = review_data.get("summary", "No summary provided.")
    if isinstance(summary, list):
        summary = "\n".join([f"* {item}" for item in summary])

    issues = review_data.get("review_report") or []
    if not isinstance(issues, list):
        st.error("Invalid review report format.")
        return
    
    full_code = review_data.get("full_corrected_code")
    
    st.subheader("Review Summary")
    st.markdown(summary)

    if not issues:
        st.success("No Specific Issues Found")
        return

    critical_issues = [i for i in issues if i.get("severity") == "CRITICAL"]
    major_issues = [i for i in issues if i.get("severity") == "MAJOR"]
    minor_issues = [i for i in issues if i.get("severity") == "MINOR"]

    st.markdown("---")
    
    if critical_issues:
        with st.expander(f"Critical Issues ({len(critical_issues)})", expanded=True):
            for issue in critical_issues:
                display_issue(issue)
    
    if major_issues:
        with st.expander(f"Major Issues ({len(major_issues)})"):
            for issue in major_issues:
                display_issue(issue)

    if minor_issues:
        with st.expander(f"Minor Issues ({len(minor_issues)})"):
            for issue in minor_issues:
                display_issue(issue)

    if full_code:
        st.markdown("---")
        st.subheader("Complete Corrected Code")
        st.info("Use the copy icon in the top right of the code block below to copy the entire corrected file.")
        st.code(full_code, language='python')

def display_issue(issue):
    st.markdown(f"**{issue.get('issue_type', 'General')}** at `{issue.get('file_path', 'N/A')}` (Lines: {issue.get('start_line', '?')}-{issue.get('end_line', '?')})")
    st.markdown(f"**Description:** {issue.get('description', 'No description provided.')}")
    if issue.get('fix_suggestion_code'):
        st.markdown("**Suggestion:**")
        st.code(issue['fix_suggestion_code'], language='python')
    st.markdown("---")

st.title("Linter.ai")

tab1, tab2 = st.tabs(["Perform Review", "Dashboard"])

with tab1:
    st.markdown("#### How it Works")
    st.markdown("""
    1.  **Select Input Method**: Choose to analyze code from a Pull Request or paste it directly.
    2.  **Configure Analysis**: Select a fast 'Quick Scan' or a 'Deep Analysis' with specialized agents.
    3.  **Generate Review**: The AI will analyze the code and provide a detailed report, including a fully corrected version.
    """)
    st.divider()

    input_method = st.radio("Input Method", ["From Pull Request", "Paste Code Directly"], horizontal=True)
    st.divider()

    if input_method == "From Pull Request":
        st.markdown("#### Analysis Configuration")
        mode = st.radio("Analysis Mode", ["Quick Scan", "Deep Analysis"], horizontal=True, key="pr_mode")
        st.divider()

        with st.form("pr_form"):
            st.markdown("#### Target Repository")
            platform = st.selectbox("Platform", ["GitHub", "GitLab"])
            repo_owner = st.text_input("Repository Owner")
            repo_name = st.text_input("Repository Name")
            pr_id = st.number_input("PR Number", 1, step=1)

            agents = []
            if mode == "Deep Analysis":
                st.markdown("#### Specialist Agents")
                all_agents = ["Security", "Performance", "Readability", "Documentation", "Error Handling"]
                selected_agents = []
                for agent in all_agents:
                    if st.checkbox(agent, value=True, key=f"pr_{agent}"):
                        selected_agents.append(agent)
                agents = selected_agents

            submitted = st.form_submit_button("Generate Review", use_container_width=True)

        if submitted:
            if mode == "Deep Analysis" and not agents:
                st.error("Please select at least one specialist agent.")
            else:
                pr_details = fetch_pr_data(repo_owner, repo_name, pr_id, platform)
                pr_diff = fetch_pr_diff(repo_owner, repo_name, pr_id, platform)

                if pr_details and pr_diff:
                    title = pr_details.get('title', '')
                    body = pr_details.get('body') or pr_details.get('description', '')
                    
                    if mode == "Quick Scan":
                        review_data = generate_quick_review(title, body, pr_diff)
                    else:
                        review_data = generate_deep_review(title, body, pr_diff, agents)
                    
                    if review_data:
                        st.divider()
                        display_review_report(review_data)
                        save_review(review_data, repo_owner, repo_name, pr_id)
                    else:
                        st.error("Failed to get a review from the AI engine.")

    else:
        st.markdown("#### Analysis Configuration")
        mode = st.radio("Analysis Mode", ["Quick Scan", "Deep Analysis"], horizontal=True, key="code_mode")
        st.divider()

        with st.form("code_form"):
            st.markdown("#### Code Input")
            code_to_review = st.text_area("Code to Review", height=300, placeholder="Paste Your Code Snippet Here")
            
            agents = []
            if mode == "Deep Analysis":
                st.markdown("#### Specialist Agents")
                all_agents = ["Security", "Performance", "Readability", "Documentation", "Error Handling"]
                selected_agents = []
                for agent in all_agents:
                    if st.checkbox(agent, value=True, key=f"code_{agent}"):
                        selected_agents.append(agent)
                agents = selected_agents

            submitted = st.form_submit_button("Generate Review", use_container_width=True)

        if submitted:
            if not code_to_review.strip():
                st.error("Please Paste Code To Review.")
            elif mode == "Deep Analysis" and not agents:
                st.error("Please select at least one specialist agent.")
            else:
                with st.spinner("Analyzing Code"):
                    title = "Direct Code Submission"
                    body = "A code snippet submitted for direct review."
                    
                    if mode == "Quick Scan":
                        review_data = generate_quick_review(title, body, code_to_review)
                    else:
                        review_data = generate_deep_review(title, body, code_to_review, agents)

                if review_data:
                    st.divider()
                    display_review_report(review_data)

with tab2:
    st.header("Code Quality Dashboard")
    all_reviews = load_reviews()

    if not all_reviews:
        st.warning("No review data found. Perform a review from a Pull Request to see the dashboard.")
    else:
        df = pd.DataFrame(all_reviews)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df.dropna(subset=['timestamp'], inplace=True)

        df['pr_link'] = df.apply(lambda row: f"{row['owner']}/{row['repo']}#{row['pr_number']}", axis=1)
        
        df['health_score'] = df['review_data'].apply(calculate_health_score)

        st.subheader("Health Score Over Time")
        if not df.empty:
            st.line_chart(df.set_index('timestamp')['health_score'])

        st.subheader("Common Issue Categories")
        all_issues = [issue for review in df['review_data'] for issue in review.get('review_report', [])]
        if all_issues:
            issue_counts = pd.Series([i.get('issue_type', 'Unknown') for i in all_issues]).value_counts()
            st.bar_chart(issue_counts)
        else:
            st.info("No issues have been logged yet.")

        st.subheader("Review History")
        st.dataframe(df[['timestamp', 'pr_link', 'health_score']], use_container_width=True)

    if st.button("Clear Dashboard Data", help="This will permanently delete all saved review history."):
        if os.path.exists(REVIEWS_FILE):
            os.remove(REVIEWS_FILE)
            st.success("Dashboard data cleared successfully! Reloading...")
            st.rerun()
        else:
            st.warning("No data to clear.")
