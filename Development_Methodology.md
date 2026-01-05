# Development Methodology & Lessons Learned  

## The Golden Rules 

### 1 NO CODE until we're 100% certain of what we're building.

We follow a deliberate analysis-first approach: understand , plan, agree, implement, validate.

### 2 NO analysis at a bulk. NO analizing a whole week of work in one take. ALWAYS analize goal by goal, step by step.

---

## Our Collaborative Workflow

### Phase 1: Understand the Problem

Before any design or code:

- Identify the actual issue (not assumptions).
- Look at real examples from the data.
- Understand the scope and scale.
- Ask clarifying questions.
- Document what we know vs. what we need to find out.

### Phase 2: Explore Options Together

- Brainstorm multiple approaches.
- Discuss pros and cons of each.
- Consider downstream impacts (Week 3, 4, 5+).
- Evaluate trade-offs honestly.
- Ask "What could go wrong?"

### Phase 3: Make Informed Decisions

- Choose the approach that best fits:
  - Current needs
  - Future extensibility
  - Data quality principles
  - Maintainability
- Get explicit agreement before proceeding.
- Document the "why" behind the decision.

### Phase 4: Code the Agreed Solution

- ONLY NOW do we write code.
- Implement exactly what was agreed.
- Follow the design decisions made.
- No surprises or scope creep.
- We reuse as much code as possible from other projects, we do no reinvent the wheel

### Phase 5: Test & Iterate

- Test with real data immediately.
- Measure results (percentages, counts, examples).
- Analyze gaps systematically.
- If <100% success, return to Phase 1.
- Iterate until satisfied.


## What We Avoid:

- Coding based on assumptions  
- "Let's try this and see" without analysis  
- Fixing problems we don't understand  
- Band-aids on top of band-aids  
- Surprises in Week 5 from Week 2 decisions

---

## Communication Style

**Issue Identified:** Initial approach was overwhelming with multiple options/questions simultaneously.

**New Approach: One Step at a Time**

### Core Principles

1. **Single Question Format**
   - Ask ONE specific question at a time.
   - Wait for answer.
   - Process answer together.
   - THEN decide next step.
   - No branching scenarios until current step complete.

2. **No Option Overload**
   - Avoid: "Here are 5 options, which do you want?"
   - Instead: "I recommend Option B because X. Agree?"
   - If user disagrees, THEN discuss alternatives.

3. **Show, Don't Speculate**
   - Provide specific command to run.
   - See actual results together.
   - Make decisions based on data, not assumptions.
   - "Run this and show me what you see."

4. **Progress Over Perfection**
   - "Good enough for MVP" is a valid choice.
   - Document limitations, continue forward.
   - Can enhance later if needed.
   - Avoid rabbit holes.

5. **Respect User Autonomy**
   - Don't ask if user is tired or wants to rest.
   - User will communicate when they need a break.
   - Focus on work until user indicates otherwise.
   - Trust user to manage their own energy.

### Code Delivery Protocol

**"Let it rip" = Provide the code immediately**

- When user says "let it rip", provide complete implementation.
- No further analysis or discussion before coding.
- Assume all planning is complete.
- Deliver production-ready code.

**Before "let it rip":**
- Continue analysis and discussion.
- Propose solutions, discuss trade-offs.
- Get explicit agreement on approach.
- ONLY code after "let it rip" or equivalent confirmation.


### Coding Best Practices

### 1. Separation of Concerns Architecture

**Principle:** Each script does ONE thing well.

### 2. Centralized Configuration is NON-NEGOTIABLE

ALL settings in `config.py`, all paths in `paths.py`:

Never hardcode. Check `config.py` and `paths.py` FIRST before writing code.

We reuse as much as possible file utils, statistics, embbeding code, colelction, handling, etc.
