"""Task catalog and free-text task matching.

TASK_CATALOG is the searchable list on the homescreen. Each entry maps a common
task to one of the task types in scoring.TASK_TYPES. Add rows freely.

match_task() handles anything the user types: it scores the text against
keywords for each task type and also looks for the closest catalog task.
"""
import difflib
import re

HARD = "Hard stuff (math, coding)"
CHAT = "Chatbot / needs to be fast"
BUDGET = "On a budget"
DEFAULT = "Default"

# (category, task, task type)
TASK_CATALOG = [
    # Coding
    ("Coding", "Write a function from a description", HARD),
    ("Coding", "Write code that passes unit tests", HARD),
    ("Coding", "Debug an error or stack trace", HARD),
    ("Coding", "Refactor or clean up existing code", HARD),
    ("Coding", "Write unit tests for my code", HARD),
    ("Coding", "Explain what a piece of code does", HARD),
    ("Coding", "Convert code from one language to another", HARD),
    ("Coding", "Write SQL queries", HARD),
    ("Coding", "Write a data-cleaning script in Python", HARD),
    ("Coding", "Solve LeetCode / interview problems", HARD),
    ("Coding", "Review a pull request for bugs", HARD),
    ("Coding", "Build a small web app or API endpoint", HARD),
    ("Coding", "Help with Java programming homework", HARD),
    ("Coding", "Write shell / bash scripts", HARD),
    # Math & reasoning
    ("Math & reasoning", "Solve math word problems", HARD),
    ("Math & reasoning", "Step-by-step algebra or calculus help", HARD),
    ("Math & reasoning", "Linear algebra homework (matrices, eigenvalues)", HARD),
    ("Math & reasoning", "Check a proof or derivation", HARD),
    ("Math & reasoning", "Statistics and probability questions", HARD),
    ("Math & reasoning", "Physics or engineering calculations", HARD),
    ("Math & reasoning", "Logic puzzles and multi-step reasoning", HARD),
    ("Math & reasoning", "Financial calculations (interest, budgets, forecasts)", HARD),
    ("Math & reasoning", "Analyze numbers in a spreadsheet", HARD),
    # Chat & support
    ("Chat & support", "Customer support chatbot", CHAT),
    ("Chat & support", "FAQ bot for a website", CHAT),
    ("Chat & support", "Course or class Q&A assistant", CHAT),
    ("Chat & support", "Real-time voice or chat assistant", CHAT),
    ("Chat & support", "Help desk / IT support replies", CHAT),
    ("Chat & support", "Booking or scheduling assistant", CHAT),
    ("Chat & support", "Onboarding guide for new users", CHAT),
    ("Chat & support", "Quick autocomplete or suggested replies", CHAT),
    ("Chat & support", "Discord or Slack bot", CHAT),
    ("Chat & support", "Language practice conversation partner", CHAT),
    # Writing & instructions
    ("Writing & instructions", "Draft professional emails", DEFAULT),
    ("Writing & instructions", "Rewrite text in a specific tone or style", DEFAULT),
    ("Writing & instructions", "Follow a strict output format (JSON, bullet list, word limit)", DEFAULT),
    ("Writing & instructions", "Write a cover letter or resume bullets", DEFAULT),
    ("Writing & instructions", "Write blog posts or articles", DEFAULT),
    ("Writing & instructions", "Proofread and fix grammar", DEFAULT),
    ("Writing & instructions", "Create social media captions", DEFAULT),
    ("Writing & instructions", "Write product descriptions", DEFAULT),
    ("Writing & instructions", "Translate text between languages", DEFAULT),
    ("Writing & instructions", "Fill in a template or form", DEFAULT),
    # Knowledge & study
    ("Knowledge & study", "Answer general knowledge questions", DEFAULT),
    ("Knowledge & study", "Explain a concept simply", DEFAULT),
    ("Knowledge & study", "Make flashcards or quiz questions", DEFAULT),
    ("Knowledge & study", "Study help for exams", DEFAULT),
    ("Knowledge & study", "Summarize a lecture or article", DEFAULT),
    ("Knowledge & study", "Research a topic and give an overview", DEFAULT),
    ("Knowledge & study", "Answer science or history questions", DEFAULT),
    ("Knowledge & study", "Compare options and give pros and cons", DEFAULT),
    # High volume / budget
    ("High volume / budget", "Classify or tag thousands of items", BUDGET),
    ("High volume / budget", "Extract fields from many documents or invoices", BUDGET),
    ("High volume / budget", "Summarize a large batch of emails or reviews", BUDGET),
    ("High volume / budget", "Sentiment analysis on customer feedback", BUDGET),
    ("High volume / budget", "Clean up or label a dataset", BUDGET),
    ("High volume / budget", "Generate many short descriptions in bulk", BUDGET),
    ("High volume / budget", "Moderate or filter user comments", BUDGET),
    ("High volume / budget", "Turn messy text into structured data", BUDGET),
    ("High volume / budget", "Translate a large set of short strings", BUDGET),
    ("High volume / budget", "Student side project with no budget", BUDGET),
]

CATALOG_LABELS = [f"{cat} — {task}" for cat, task, _ in TASK_CATALOG]
CATALOG_TYPE = {f"{cat} — {task}": tt for cat, task, tt in TASK_CATALOG}

KEYWORDS = {
    HARD: ["math", "code", "coding", "program", "python", "java", "javascript", "c++", "function",
           "bug", "debug", "error", "stack trace", "refactor", "algorithm", "leetcode", "equation",
           "calculat", "solve", "proof", "derivation", "sql", "query", "script", "unit test", "test case",
           "homework", "algebra", "calculus", "linear algebra", "matrix", "matrices", "eigen", "statistic",
           "probability", "physics", "engineering", "logic", "reasoning", "puzzle", "api", "compile",
           "numpy", "spreadsheet formula", "data analysis", "analyze data"],
    CHAT: ["chat", "chatbot", "bot", "customer", "support", "help desk", "helpdesk", "faq", "reply",
           "replies", "conversation", "real-time", "realtime", "real time", "fast", "quick", "instant",
           "assistant", "discord", "slack", "voice", "live", "respond", "users ask", "booking",
           "schedul", "onboarding", "autocomplete", "low latency", "latency"],
    BUDGET: ["cheap", "budget", "low cost", "low-cost", "inexpensive", "free", "bulk", "batch",
             "thousands", "millions", "hundreds", "large volume", "high volume", "lots of", "many",
             "save money", "classify", "tag", "label", "extract", "invoice", "sentiment", "moderate",
             "filter", "dataset", "every", "all of our"],
    DEFAULT: ["question", "fact", "explain", "knowledge", "trivia", "summar", "email", "format",
              "write", "writing", "essay", "blog", "article", "instruction", "research", "study",
              "exam", "quiz", "flashcard", "translate", "rewrite", "proofread", "grammar", "tone",
              "resume", "cover letter", "caption", "template", "history", "science", "overview"],
}


# Words that state the priority outright count double.
STRONG = {"cheap", "budget", "low cost", "low-cost", "inexpensive", "save money", "real-time", "realtime",
          "real time", "low latency", "chatbot", "math", "code", "coding", "debug"}


def match_task(text: str):
    """Return (task_type or None, reason string) for free text."""
    t = (text or "").lower().strip()
    if not t:
        return None, ""
    if t in (lbl.lower() for lbl in CATALOG_LABELS):
        lbl = next(l for l in CATALOG_LABELS if l.lower() == t)
        return CATALOG_TYPE[lbl], f"you picked “{lbl.split(' — ', 1)[1]}”"

    hits = {}
    for tt, kws in KEYWORDS.items():
        found = []
        for k in kws:
            m = re.search(r"\b" + re.escape(k) + r"[\w-]*", t)
            if m and m.group(0) not in [w for w, _ in found]:
                found.append((m.group(0), k))
        hits[tt] = found
    score = {tt: sum(2 if k in STRONG else 1 for _, k in h) for tt, h in hits.items()}

    # Closest catalog task as a tie-breaker / fallback.
    tasks_lower = [task.lower() for _, task, _ in TASK_CATALOG]
    close = difflib.get_close_matches(t, tasks_lower, n=1, cutoff=0.45)
    if close:
        i = tasks_lower.index(close[0])
        score[TASK_CATALOG[i][2]] += 0.75

    best = max(score, key=score.get)
    if score[best] == 0:
        return None, ""
    words = [w for w, _ in hits[best]][:4]
    if words:
        reason = "it mentions " + ", ".join(f"“{w}”" for w in words)
    else:
        reason = f"it's similar to “{TASK_CATALOG[tasks_lower.index(close[0])][1]}”"
    return best, reason
