# MindGuard Ethical Pilot Framework
## Deploying a Mental Health Screening Tool in School Settings

> **Version:** 1.0 | **Status:** Pre-pilot planning  
> **Target:** Under-resourced high school settings with school counselor partnership

---

## ⚠️ Critical Preamble

This framework describes how to **ethically** pilot a mental health *screening* tool in a school setting. This is NOT a diagnostic system. All flags MUST be reviewed by trained school counselors. The tool is designed to augment — never replace — human professional judgment.

**Before any pilot deployment, the following MUST be completed:**
- [ ] IRB (Institutional Review Board) approval or equivalent ethical review
- [ ] School district approval and legal review
- [ ] Parental/guardian informed consent (for minors)
- [ ] Student assent process
- [ ] Counselor training on tool limitations
- [ ] Crisis protocol integration
- [ ] Data privacy impact assessment

---

## 1. Consent Process

### 1.1 Parental/Guardian Consent
- **Opt-in only** — parents must actively agree, not passively accept
- Consent form must explain in plain language:
  - What the tool does and does NOT do
  - That it is a screening aid, not a diagnosis
  - That no personally identifiable text is stored
  - That the tool flags text for HUMAN review only
  - Their right to withdraw at any time
  - Contact information for questions
- Available in multiple languages as needed
- Reviewed by a lawyer familiar with FERPA/COPPA

### 1.2 Student Assent
- Age-appropriate explanation of the tool
- Students understand their anonymous responses may be screened
- Students can opt out at any time without consequence
- Clear communication that this is about support, not surveillance

### 1.3 Counselor Agreement
- Counselors acknowledge the tool's limitations
- Agree to review ALL flagged responses
- Understand false positive/negative rates
- Commit to following existing crisis protocols

---

## 2. Data Anonymization

### 2.1 Design Principles
- **No PII collection**: The system processes text only — no names, IDs, or demographics
- **No text storage**: Input text is processed in-memory and immediately discarded
- **No logging**: User inputs are never written to disk, logs, or databases
- **Anonymous check-ins**: Students submit via anonymous forms (no login required)
- **Aggregated reporting only**: School reports show only aggregate statistics

### 2.2 Technical Safeguards
| Safeguard | Implementation |
|-----------|---------------|
| No input logging | `log_inputs: false` in config; verified in code review |
| In-memory processing | Text processed, result returned, text deleted |
| No model memorization | Model is frozen during inference; no online learning |
| No IP tracking | Server configured without access logs |
| Encrypted transit | HTTPS required for any network communication |

### 2.3 What Data IS Collected (Aggregated Only)
- Number of check-ins per week (not per student)
- Distribution of severity predictions (aggregate)
- System performance metrics (latency, errors)
- Counselor feedback on prediction accuracy (no student text)

---

## 3. Counselor Workflow

### 3.1 Daily Workflow
```
1. Students complete anonymous weekly check-in (3-5 sentences)
2. Text is screened by MindGuard
3. Results are categorized into severity bands:
   
   🟢 No Concern → No action needed
   🟡 Mild       → Added to counselor's awareness list
   🟠 Moderate   → Counselor follow-up within 48 hours
   🔴 Severe     → Immediate review; activate crisis protocol
   
4. Counselor reviews ALL flagged responses (Moderate + Severe)
5. Counselor makes independent clinical judgment
6. Counselor documents decision (not the text)
```

### 3.2 Key Principles
- **The tool SUGGESTS; the counselor DECIDES**
- Never rely on the tool alone for crisis determination
- Consider the tool's output as ONE data point among many
- Follow existing school crisis protocols for Severe flags
- Document your clinical reasoning, not the tool's output

### 3.3 Handling False Positives
- A false positive means the tool flagged text as concerning when it wasn't
- **This is the PREFERRED error** — better safe than sorry
- Counselor reviews and dismisses false positives
- Tracked in aggregate to monitor model performance

### 3.4 Handling False Negatives
- A false negative means the tool missed a concerning response
- **This is the CRITICAL error** — a student in crisis was not flagged
- Mitigated by: counselors also reviewing a random sample of "No Concern" texts
- All crisis protocols remain independent of the tool

---

## 4. Pilot Study Design

### 4.1 Phase 1: Shadow Mode (Weeks 1-4)
- Tool runs alongside existing processes
- Results shown ONLY to counselors, not acted upon
- Counselors compare tool output to their own assessment
- Goal: Measure agreement rate, identify systematic errors
- **No student outcomes depend on the tool**

### 4.2 Phase 2: Advisory Mode (Weeks 5-12)
- Tool provides suggestions alongside counselor's workflow
- Counselor always makes the final decision
- Track: time-to-flag, counselor agreement rate, false negative rate
- Weekly calibration meetings with counseling team

### 4.3 Phase 3: Evaluation (Weeks 13-16)
- Analyze aggregate outcomes:
  - Did flagging speed improve?
  - Were there fewer missed check-ins?
  - How did counselors rate the tool's usefulness?
- Student satisfaction survey (anonymous)
- Counselor workload assessment

### 4.4 Sample Size
- Minimum: 100 anonymous check-ins for meaningful statistics
- Target: 500+ check-ins over the pilot period
- At least 3 counselors participating for inter-rater comparison

---

## 5. Success Metrics

| Metric | Target | Why |
|--------|--------|-----|
| Crisis Recall | ≥ 95% | Must catch nearly all severe cases |
| False Positive Rate | ≤ 30% | Manageable counselor workload |
| Counselor Agreement | ≥ 70% | Tool is clinically meaningful |
| Time-to-Flag | < 1 min | Faster than manual review |
| Counselor Satisfaction | ≥ 4/5 | Tool must be useful, not burdensome |
| Student Opt-out Rate | < 10% | Students trust the process |
| Bias Audit | < 10% flip rate | Fair across demographics |

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| False negative (missed crisis) | Random sampling of "No Concern" texts; independent crisis protocols |
| Over-reliance on tool | Training emphasizes tool limitations; shadow mode first |
| Privacy breach | No text storage; anonymous input; encryption |
| Stigmatization | Anonymous process; severity bands, not binary labels |
| Demographic bias | Bias audit before deployment; ongoing monitoring |
| Tool failure | All crisis protocols independent of tool; manual backup |

---

## 7. Ethical Review Checklist

- [ ] IRB or ethics board approval obtained
- [ ] FERPA compliance verified
- [ ] Parental consent process approved by legal counsel
- [ ] Student assent form reviewed by age-appropriate expert
- [ ] Counselor training completed (minimum 2 hours)
- [ ] Crisis protocol integration verified
- [ ] Data privacy impact assessment completed
- [ ] Bias audit passed (< 10% flip rate)
- [ ] False negative analysis documented
- [ ] Opt-out mechanism tested
- [ ] Aggregate reporting system tested
- [ ] Emergency contact protocols verified
- [ ] Weekly review meeting schedule established

---

## 8. Termination Criteria

The pilot MUST be stopped immediately if:
1. Any student's safety is compromised due to a tool error
2. Counselors report the tool is causing more harm than benefit
3. Bias audit reveals > 15% prediction flip rate for any demographic
4. False negative rate exceeds 10% on severe crisis cases
5. Student opt-out rate exceeds 25%
6. Data privacy breach occurs

---

## 9. Post-Pilot

### 9.1 Deliverables
- Aggregate performance report (no student-level data)
- Counselor feedback summary
- Updated bias audit
- Recommendations for improvement
- Research paper contribution

### 9.2 Data Retention
- **Student text**: NOT retained (never stored in the first place)
- **Aggregate statistics**: Retained for research (anonymized)
- **Counselor feedback**: Retained with counselor consent
- **Model weights**: Retained for future research

---

*This framework was designed with input from AI ethics guidelines (EU AI Act, 
OECD AI Principles) and mental health research ethics (APA, WHO). It should 
be reviewed by your school's legal counsel, ethics board, and clinical 
supervisor before implementation.*
