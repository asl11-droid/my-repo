"""Task catalog and free-text task matching.

Every task maps to one of the four MVP categories (costmodel.CATEGORIES):
coding (HumanEval+), knowledge (MMLU-Pro), math (GSM8K), ifeval (IFEval),
plus a suggested waiting preset (how much of the wait the user actually loses).
Add rows to TASK_CATALOG freely.
"""
import difflib
import re

I, S, B = ("Interactive (chat, pair coding)", "Semi-interactive (I switch tabs)", "Batch / background")

# (group, task, category, waiting preset)
TASK_CATALOG = [
    # Coding
    ("Coding", "Write a function from a description", "coding", S),
    ("Coding", "Write code that passes unit tests", "coding", S),
    ("Coding", "Debug an error or stack trace", "coding", I),
    ("Coding", "Refactor or clean up existing code", "coding", S),
    ("Coding", "Write unit tests for my code", "coding", S),
    ("Coding", "Explain what a piece of code does", "coding", I),
    ("Coding", "Convert code from one language to another", "coding", S),
    ("Coding", "Write SQL queries", "coding", I),
    ("Coding", "Write a data-cleaning script in Python", "coding", S),
    ("Coding", "Solve LeetCode / interview problems", "coding", I),
    ("Coding", "Review a pull request for bugs", "coding", B),
    ("Coding", "Build a small web app or API endpoint", "coding", S),
    ("Coding", "Help with Java programming homework", "coding", I),
    ("Coding", "Pair-program with an AI in my editor", "coding", I),
    ("Coding", "Write shell / bash scripts", "coding", S),
    # Math
    ("Math", "Solve math word problems", "math", I),
    ("Math", "Step-by-step algebra or calculus help", "math", I),
    ("Math", "Linear algebra homework (matrices, eigenvalues)", "math", I),
    ("Math", "Check my homework answers", "math", S),
    ("Math", "Statistics and probability questions", "math", I),
    ("Math", "Physics or engineering calculations", "math", S),
    ("Math", "Logic puzzles and multi-step reasoning", "math", I),
    ("Math", "Financial calculations (interest, budgets, forecasts)", "math", S),
    ("Math", "Analyze numbers in a spreadsheet", "math", S),
    ("Math", "Grade a batch of math answers", "math", B),
    # Instruction following
    ("Instructions & writing", "Customer support chatbot that follows our policies", "ifeval", I),
    ("Instructions & writing", "Draft professional emails", "ifeval", S),
    ("Instructions & writing", "Rewrite text in a specific tone or style", "ifeval", S),
    ("Instructions & writing", "Follow a strict output format (JSON, bullet list, word limit)", "ifeval", S),
    ("Instructions & writing", "Write a cover letter or resume bullets", "ifeval", S),
    ("Instructions & writing", "Proofread and fix grammar", "ifeval", S),
    ("Instructions & writing", "Create social media captions", "ifeval", S),
    ("Instructions & writing", "Fill in a template or form", "ifeval", S),
    ("Instructions & writing", "Booking or scheduling assistant", "ifeval", I),
    ("Instructions & writing", "Discord or Slack bot", "ifeval", I),
    ("Instructions & writing", "Quick autocomplete or suggested replies", "ifeval", I),
    ("Instructions & writing", "Classify or tag thousands of items", "ifeval", B),
    ("Instructions & writing", "Extract fields from many documents or invoices", "ifeval", B),
    ("Instructions & writing", "Summarize a large batch of emails or reviews", "ifeval", B),
    ("Instructions & writing", "Turn messy text into structured data", "ifeval", B),
    ("Instructions & writing", "Moderate or filter user comments", "ifeval", B),
    ("Instructions & writing", "Translate a large set of short strings", "ifeval", B),
    # General knowledge
    ("General knowledge", "Answer general knowledge questions", "knowledge", I),
    ("General knowledge", "FAQ bot for a website", "knowledge", I),
    ("General knowledge", "Course or class Q&A assistant", "knowledge", I),
    ("General knowledge", "Help desk / IT support answers", "knowledge", I),
    ("General knowledge", "Explain a concept simply", "knowledge", I),
    ("General knowledge", "Make flashcards or quiz questions", "knowledge", S),
    ("General knowledge", "Study help for exams", "knowledge", I),
    ("General knowledge", "Summarize a lecture or article", "knowledge", S),
    ("General knowledge", "Research a topic and give an overview", "knowledge", S),
    ("General knowledge", "Answer science or history questions", "knowledge", I),
    ("General knowledge", "Compare options and give pros and cons", "knowledge", S),
    ("General knowledge", "Sentiment analysis on customer feedback", "knowledge", B),
    ("General knowledge", "Language practice conversation partner", "knowledge", I),
    ("General knowledge", "Medical or legal background reading (not advice)", "knowledge", S),
    ("General knowledge", "Trivia or quiz game host", "knowledge", I),
    ("General knowledge", "Fact-check a draft", "knowledge", S),
    ("General knowledge", "Write blog posts or articles", "knowledge", S),
    ("General knowledge", "Write product descriptions in bulk", "knowledge", B),
    ("General knowledge", "Onboarding guide for new users", "knowledge", I),
]

CATALOG_LABELS = [f"{g} — {t}" for g, t, _, _ in TASK_CATALOG]
CATALOG = {f"{g} — {t}": (t, c, w) for g, t, c, w in TASK_CATALOG}

KEYWORDS = {
    "coding": ["code", "coding", "program", "python", "java", "javascript", "typescript", "c++", "rust",
               "function", "bug", "debug", "stack trace", "refactor", "algorithm", "leetcode", "sql",
               "script", "unit test", "test case", "api", "compile", "repo", "pull request", "web app",
               "frontend", "backend", "html", "css", "regex"],
    "math": ["math", "equation", "calculat", "solve", "proof", "derivation", "algebra", "calculus",
             "linear algebra", "matrix", "matrices", "eigen", "statistic", "probability", "physics",
             "engineering", "logic", "reasoning", "puzzle", "numbers", "arithmetic", "geometry",
             "interest rate", "forecast", "homework answers", "word problem"],
    "ifeval": ["format", "instruction", "json", "template", "email", "rewrite", "tone", "style", "proofread",
               "grammar", "caption", "resume", "cover letter", "word limit", "bullet", "classify", "tag",
               "label", "extract", "invoice", "structured", "moderate", "filter", "translate", "policy",
               "policies", "customer", "booking", "schedul", "autocomplete", "form", "follow"],
    "knowledge": ["question", "fact", "explain", "knowledge", "trivia", "summar", "research", "study",
                  "exam", "quiz", "flashcard", "history", "science", "overview", "faq", "help desk",
                  "lecture", "article", "learn", "teach", "course", "syllabus", "biology", "chemistry",
                  "economics", "blog", "sentiment", "compare", "pros and cons"],
}
STRONG = {"code", "coding", "debug", "math", "json", "format", "trivia", "faq", "sql", "proof"}

WAIT_WORDS = {
    I: ["chat", "chatbot", "bot", "real-time", "realtime", "real time", "live", "instant", "quick",
        "fast", "interactive", "voice", "assistant", "pair", "while i wait", "users ask", "customer"],
    B: ["thousands", "millions", "hundreds", "bulk", "batch", "overnight", "background", "every",
        "all of our", "large volume", "high volume", "dataset", "pipeline", "nightly"],
}


def _hits(t, words):
    found = []
    for k in words:
        m = re.search(r"\b" + re.escape(k) + r"[\w-]*", t)
        if m and m.group(0) not in [w for w, _ in found]:
            found.append((m.group(0), k))
    return found


def guess_wait(text):
    t = (text or "").lower()
    big = any(int(n.replace(",", "")) >= 1000 for n in re.findall(r"\d[\d,]*", t) if n.replace(",", "").isdigit())
    if big or _hits(t, WAIT_WORDS[B]):
        return B
    if _hits(t, WAIT_WORDS[I]):
        return I
    return S


def match_task(text: str):
    """Return (category or None, waiting preset, reason) for free text."""
    t = (text or "").lower().strip()
    if not t:
        return None, S, ""
    hits = {c: _hits(t, kws) for c, kws in KEYWORDS.items()}
    score = {c: sum(2 if k in STRONG else 1 for _, k in h) for c, h in hits.items()}
    if sum(score.values()) == 0:          # no real signal: ask the user instead of guessing
        return None, guess_wait(t), ""
    tasks_lower = [task.lower() for _, task, _, _ in TASK_CATALOG]
    close = difflib.get_close_matches(t, tasks_lower, n=1, cutoff=0.45)
    if close:
        score[TASK_CATALOG[tasks_lower.index(close[0])][2]] += 0.75
    best = max(score, key=score.get)
    if score[best] == 0:
        return None, guess_wait(t), ""
    words = [w for w, _ in hits[best]][:4]
    reason = ("it mentions " + ", ".join(f"“{w}”" for w in words)) if words else \
             f"it's similar to “{TASK_CATALOG[tasks_lower.index(close[0])][1]}”"
    return best, guess_wait(t), reason
