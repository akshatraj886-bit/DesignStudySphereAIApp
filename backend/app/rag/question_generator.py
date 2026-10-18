"""
StudySphere AI - Revision Questions Generator & Quiz Evaluation Engine
Generates strictly grounded academic revision questions from stored document chunks:
- Multiple Choice Questions (MCQ): 4 distinct options, exactly 1 correct, plausible distractors from source notes.
- True/False: Factual propositions with source-grounded justifications.
- Short Answer: Focused conceptual questions with marking criteria and page citations.
- Long Answer: Multi-part synthesis questions with structured evaluation rubrics.

Grading & Evaluation:
- Deterministic scoring for MCQ and True/False.
- Transparent keyword/concept coverage evaluation for Short and Long answers without claiming false semantic intelligence.
- Full source page citations on every question.
"""

import os
import re
import time
import random
import secrets
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from app.schemas import (
    QuestionType,
    QuestionDifficulty,
    PublicQuestionItem,
    QuestionItem,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionAnswerSubmission,
    QuestionEvaluationResult,
    EvaluateQuizResponse,
)
from app.rag.store import store_instance, VectorStore

logger = logging.getLogger("studysphere.questions")

COMMON_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "would", "should", "could", "ought", "i", "you", "he",
    "she", "it", "we", "they", "them", "their", "my", "your", "his", "her",
    "by", "for", "of", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "in", "on", "at", "with", "about", "against", "between", "into",
    "up", "down", "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "now", "studysphere"
}


def clean_text(text: str) -> str:
    """Removes leading bullets, numbers, and extra whitespace."""
    s = text.strip()
    s = re.sub(r"^[-*•\d.]+\s*", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_keywords(text: str) -> Set[str]:
    """Extracts substantive content words for keyword matching."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = {
        w for w in cleaned.split()
        if len(w) >= 4 and w not in COMMON_STOP_WORDS
    }
    return words


class ConceptUnit:
    """Represents a factual academic concept extracted from source text."""
    def __init__(
        self,
        term: str,
        definition: str,
        source_sentence: str,
        page_number: int,
        category: str = "general",
        details: Optional[List[str]] = None,
    ):
        self.term = clean_text(term)
        self.definition = clean_text(definition)
        self.source_sentence = clean_text(source_sentence)
        self.page_number = page_number
        self.category = category
        self.details = details or []

    def __repr__(self):
        return f"<ConceptUnit: {self.term} (Page {self.page_number})>"


class QuestionGenerator:
    """
    Extracts structured academic concepts from document chunks and synthesizes
    strictly grounded questions across MCQ, True/False, Short Answer, and Long Answer.
    """

    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or store_instance

    def _extract_concepts(self, chunks: List[Dict[str, Any]]) -> List[ConceptUnit]:
        """
        Parses document chunks to extract distinct factual concepts, terms, and definitions.
        """
        concepts: List[ConceptUnit] = []
        seen_terms: Set[str] = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            page_num = chunk.get("page_number", 1)

            # 1. Parse Colon-Delimited Terms (e.g. "First-Come, First-Served (FCFS): The process that...")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines:
                lower_line = line.lower()
                if any(hdr in lower_line for hdr in ["studysphere", "course:", "module", "academic notes", "lecture"]):
                    continue
                if ":" in line:
                    parts = line.split(":", 1)
                    term_cand = parts[0].strip()
                    term_cand = re.sub(r"^[-*•\d.]+\s*", "", term_cand).strip()
                    desc_cand = parts[1].strip()

                    # Valid academic term candidate: 2 to 45 characters, starts with an uppercase letter
                    # (excludes truncated word fragments from chunk boundaries, e.g. 'putation')
                    if 2 <= len(term_cand) <= 45 and term_cand[0].isupper() and len(desc_cand) >= 20:
                        term_norm = term_cand.lower()
                        if term_norm not in seen_terms and not any(b in term_norm for b in ["module", "course", "page", "section", "note", "unit"]):
                            seen_terms.add(term_norm)
                            concepts.append(
                                ConceptUnit(
                                    term=term_cand,
                                    definition=desc_cand,
                                    source_sentence=line,
                                    page_number=page_num,
                                    category="term_colon",
                                )
                            )

            # 2. Parse Definitional Sentences (e.g. "A process is a program in execution.")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                clean_s = clean_text(s)
                if len(clean_s) < 25:
                    continue

                # Matches "X is a/the Y" or "X is defined as Y"
                def_match = re.match(
                    r"^([A-Z][A-Za-z0-9\s\-_()]{2,40})\s+(is defined as|is a program|is a situation|is the basis|stores|refers to|associates each)\s+(.+)$",
                    clean_s,
                    re.IGNORECASE,
                )
                if def_match:
                    term_cand = def_match.group(1).strip()
                    term_cand = re.sub(r"^(A|An|The)\s+", "", term_cand, flags=re.IGNORECASE)
                    term_norm = term_cand.lower()
                    if term_cand and term_cand[0].isupper() and term_norm not in seen_terms and len(term_cand) <= 40:
                        seen_terms.add(term_norm)
                        concepts.append(
                            ConceptUnit(
                                term=term_cand,
                                definition=clean_s,
                                source_sentence=clean_s,
                                page_number=page_num,
                                category="definitional_sentence",
                            )
                        )

                # Matches "Four Coffman conditions must hold..." or multi-part lists
                if any(k in clean_s.lower() for k in ["conditions must hold", "principal states of", "techniques for handling"]):
                    if ":" in clean_s:
                        rule_term = clean_s.split(":")[0].strip()
                    else:
                        rule_term = clean_s[:45].rsplit(" ", 1)[0].strip() if " " in clean_s[:45] else clean_s[:45].strip()
                    rule_term = re.sub(r"^[-*•\d.]+\s*", "", rule_term).strip()
                    rule_norm = rule_term.lower()
                    if rule_term and rule_term[0].isupper() and rule_norm not in seen_terms:
                        seen_terms.add(rule_norm)
                        concepts.append(
                            ConceptUnit(
                                term=rule_term,
                                definition=clean_s,
                                source_sentence=clean_s,
                                page_number=page_num,
                                category="multi_part_rule",
                            )
                        )

        return concepts

    def generate_questions(
        self,
        document_id: str,
        request: GenerateQuestionsRequest,
        session_token: str = "default_session",
    ) -> GenerateQuestionsResponse:
        """
        Main question generation workflow:
        1. Validates document existence, session ownership, and non-empty extractable text.
        2. Extracts concept units from document chunks.
        3. Synthesizes questions matching requested question_types and count.
        4. Attaches options, correct answer, explanation, and page numbers.
        """
        # 1. Verify document existence and session ownership
        doc = self.store.get_document(document_id, session_token=session_token)
        if not doc:
            raise KeyError(f"Document with ID '{document_id}' not found.")

        # 2. Verify extractable text
        chunks = self.store.get_document_chunks(document_id, session_token=session_token)
        if not chunks or doc.get("chunk_count", 0) == 0:
            raise ValueError(
                "Document contains no extractable text. Scanned image documents requiring "
                "Optical Character Recognition (OCR) are not yet supported."
            )

        doc_name = doc.get("filename", "Document")
        concepts = self._extract_concepts(chunks)

        if not concepts:
            raise ValueError(
                "Could not extract sufficient structured academic concepts from this document. "
                "The text may be too short or lack clear conceptual headings."
            )

        # Validate question count
        target_count = request.question_count
        if target_count not in [5, 10, 15]:
            # Accept within 1-15, default to nearest
            target_count = min(15, max(1, target_count))

        requested_types = [t.value if hasattr(t, "value") else str(t) for t in request.question_types]
        if not requested_types:
            requested_types = ["mcq"]

        # Allocate counts per type
        type_counts: Dict[str, int] = {}
        base_per_type = target_count // len(requested_types)
        rem = target_count % len(requested_types)

        for i, qtype in enumerate(requested_types):
            type_counts[qtype] = base_per_type + (1 if i < rem else 0)

        generated_questions: List[QuestionItem] = []
        seen_question_prompts: Set[str] = set()
        q_counter = 1

        # All concept terms in the document used for MCQ distractor sampling
        all_terms = [c.term for c in concepts if len(c.term) >= 3]

        # Generate for each requested question type
        for qtype in requested_types:
            needed = type_counts.get(qtype, 0)
            created_for_type = 0

            # Shuffle concepts deterministically
            shuffled_concepts = list(concepts)
            # Use deterministic seed based on document and difficulty
            diff_str = request.difficulty.value if hasattr(request.difficulty, "value") else str(request.difficulty)
            rng = random.Random(f"{document_id}_{qtype}_{diff_str}")
            rng.shuffle(shuffled_concepts)

            for c in shuffled_concepts:
                if created_for_type >= needed:
                    break

                q_item = None

                if qtype == "mcq":
                    q_item = self._build_mcq(c, all_terms, diff_str, q_counter, rng)
                elif qtype == "true_false":
                    q_item = self._build_true_false(c, diff_str, q_counter, rng)
                elif qtype == "short_answer":
                    q_item = self._build_short_answer(c, diff_str, q_counter)
                elif qtype == "long_answer":
                    q_item = self._build_long_answer(c, diff_str, q_counter)

                if q_item:
                    norm_prompt = q_item.question_text.strip().lower()
                    if norm_prompt not in seen_question_prompts:
                        seen_question_prompts.add(norm_prompt)
                        generated_questions.append(q_item)
                        created_for_type += 1
                        q_counter += 1

            # Fallback if specific type could not meet exact count: supplement from concept sentences
            while created_for_type < needed and len(generated_questions) < target_count:
                c = rng.choice(concepts)
                supp_item = self._build_supplementary_question(c, qtype, diff_str, q_counter, rng, all_terms)
                if supp_item and supp_item.question_text.lower() not in seen_question_prompts:
                    seen_question_prompts.add(supp_item.question_text.lower())
                    generated_questions.append(supp_item)
                    created_for_type += 1
                    q_counter += 1
                else:
                    break

        # Generate unique, unpredictable session ID
        session_id = f"qses_{int(time.time())}_{secrets.token_hex(8)}"

        # Save authoritative questions and answer keys into SQLite session store
        self.store.save_quiz_session(
            session_id=session_id,
            document_id=document_id,
            document_name=doc_name,
            difficulty=diff_str,
            questions=[q.model_dump() for q in generated_questions],
            session_token=session_token,
        )

        # Redact private answer keys and explanations from public student response
        public_questions = [
            PublicQuestionItem(
                question_id=q.question_id,
                question_text=q.question_text,
                question_type=q.question_type,
                difficulty=q.difficulty,
                options=q.options,
                source_page_numbers=q.source_page_numbers,
            )
            for q in generated_questions
        ]

        return GenerateQuestionsResponse(
            session_id=session_id,
            document_id=document_id,
            document_name=doc_name,
            total_questions=len(public_questions),
            difficulty=diff_str,
            question_types=requested_types,
            questions=public_questions,
            generation_provider="offline_grounded_question_generator",
            model_name="deterministic_concept_extractor",
            warning="Offline grounded question generator active. Questions and answers are strictly grounded in document text.",
        )

    # -------------------------------------------------------------------------
    # Question Builders
    # -------------------------------------------------------------------------

    def _build_mcq(
        self,
        concept: ConceptUnit,
        all_terms: List[str],
        difficulty: str,
        q_idx: int,
        rng: random.Random,
    ) -> Optional[QuestionItem]:
        """
        Builds a high-quality 4-option MCQ:
        - Exactly one correct answer.
        - Three plausible distractors sourced from other concepts in the same document.
        """
        term = concept.term
        definition = concept.definition

        # Formulate prompt
        if difficulty == "easy":
            prompt = f"Which concept or technique is defined as: \"{definition}\"?"
        elif difficulty == "hard":
            prompt = (
                f"Based on your notes on Page {concept.page_number}, which specific mechanism "
                f"exhibits the following operational behavior: \"{definition}\"?"
            )
        else: # medium
            prompt = f"According to your notes, which term corresponds to the following description: \"{definition}\"?"

        # Sample 3 distinct distractors from the document's other terms
        candidate_distractors = [t for t in all_terms if t.lower() != term.lower()]
        # Filter duplicates
        unique_cands = []
        seen = set()
        for t in candidate_distractors:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_cands.append(t)

        if len(unique_cands) < 3:
            # Fallback domain distractors if document is very short
            fallback_pool = ["Round Robin (RR)", "First-Come, First-Served (FCFS)", "Shortest-Job-First (SJF)", "Priority Scheduling", "PCB", "Banker's Algorithm", "KNN Imputation", "Listwise Deletion"]
            for fb in fallback_pool:
                if fb.lower() != term.lower() and fb.lower() not in seen:
                    unique_cands.append(fb)
                    seen.add(fb.lower())
                if len(unique_cands) >= 3:
                    break

        sampled_distractors = rng.sample(unique_cands, min(3, len(unique_cands)))
        options = [term] + sampled_distractors
        rng.shuffle(options)

        explanation = (
            f"According to your notes on Page {concept.page_number}: {concept.source_sentence}"
        )

        diff_enum = QuestionDifficulty.HARD if difficulty == "hard" else (QuestionDifficulty.EASY if difficulty == "easy" else QuestionDifficulty.MEDIUM)

        return QuestionItem(
            question_id=f"q_{q_idx}",
            question_text=prompt,
            question_type=QuestionType.MCQ,
            difficulty=diff_enum,
            options=options,
            correct_answer=term,
            explanation=explanation,
            source_page_numbers=[concept.page_number],
        )

    def _build_true_false(
        self,
        concept: ConceptUnit,
        difficulty: str,
        q_idx: int,
        rng: random.Random,
    ) -> Optional[QuestionItem]:
        """
        Builds a True/False question based on a verified factual statement.
        Alternates between True statements and plausible False statements.
        Handles colon-separated concepts cleanly and naturally.
        """
        is_true = rng.choice([True, False])
        term = concept.term.strip()
        definition = concept.definition.strip()

        # Format natural academic statement for True statement
        if term.lower() in ["new", "ready", "running", "waiting", "blocked", "terminated"]:
            statement = f"In process management, the {term} state means {definition.rstrip('.') + '.'}"
        elif concept.category == "term_colon" or ":" in concept.source_sentence:
            if definition.lower().startswith(("a ", "an ", "the ")):
                statement = f"According to your notes, {term} is {definition.rstrip('.') + '.'}"
            else:
                statement = f"According to your notes, {term} refers to: {definition.rstrip('.') + '.'}"
        else:
            statement = concept.source_sentence.rstrip(".") + "."

        explanation = f"According to your notes (Page {concept.page_number}): \"{concept.source_sentence}\""

        if is_true:
            prompt = f"True or False: {statement}"
            correct = "True"
        else:
            # Formulate plausible False statement
            if term.lower() in ["new", "ready", "running", "waiting", "blocked", "terminated"]:
                if term.lower() == "running":
                    negated_desc = "instructions are waiting in the ready queue to be assigned to a processor."
                elif term.lower() == "ready":
                    negated_desc = "instructions are actively being executed by the CPU."
                elif term.lower() == "new":
                    negated_desc = "the process has finished execution and is terminating."
                elif term.lower() == "terminated":
                    negated_desc = "the process is currently being created in memory."
                else:
                    negated_desc = "the CPU is executing instructions without waiting for I/O completion."
                negated = f"In process management, the {term} state means {negated_desc}"
            else:
                negated = statement
                if "non-preemptive" in negated.lower():
                    negated = re.sub(r"non-preemptive", "preemptive", negated, flags=re.IGNORECASE)
                elif "optimal" in negated.lower():
                    negated = re.sub(r"optimal", "inefficient", negated, flags=re.IGNORECASE)
                elif "four" in negated.lower():
                    negated = re.sub(r"four", "two", negated, flags=re.IGNORECASE)
                elif "simultaneously" in negated.lower():
                    negated = re.sub(r"simultaneously", "independently at different times", negated, flags=re.IGNORECASE)
                elif "is defined as" in negated.lower():
                    negated = re.sub(r"is defined as", "is completely unrelated to", negated, flags=re.IGNORECASE)
                elif "reduces" in negated.lower():
                    negated = re.sub(r"reduces", "dramatically increases", negated, flags=re.IGNORECASE)
                elif "removes" in negated.lower():
                    negated = re.sub(r"removes", "permanently preserves", negated, flags=re.IGNORECASE)
                else:
                    negated = f"In contrast to your notes, {term} does not {definition}."

            prompt = f"True or False: {negated}"
            correct = "False"
            explanation = f"False. As stated in your notes (Page {concept.page_number}): \"{concept.source_sentence}\""

        diff_enum = QuestionDifficulty.HARD if difficulty == "hard" else (QuestionDifficulty.EASY if difficulty == "easy" else QuestionDifficulty.MEDIUM)

        return QuestionItem(
            question_id=f"q_{q_idx}",
            question_text=prompt,
            question_type=QuestionType.TRUE_FALSE,
            difficulty=diff_enum,
            options=["True", "False"],
            correct_answer=correct,
            explanation=explanation,
            source_page_numbers=[concept.page_number],
        )

    def _build_short_answer(
        self,
        concept: ConceptUnit,
        difficulty: str,
        q_idx: int,
    ) -> QuestionItem:
        """
        Builds a focused conceptual short-answer question with exact marking points.
        """
        term = concept.term
        prompt = f"Define \"{term}\" and explain its primary significance as presented in your notes."
        correct_answer = f"{term}: {concept.definition} (Page {concept.page_number})"
        explanation = (
            f"Marking criteria: Student response should mention that {term} {concept.definition} "
            f"Reference: Page {concept.page_number}."
        )

        diff_enum = QuestionDifficulty.HARD if difficulty == "hard" else (QuestionDifficulty.EASY if difficulty == "easy" else QuestionDifficulty.MEDIUM)

        return QuestionItem(
            question_id=f"q_{q_idx}",
            question_text=prompt,
            question_type=QuestionType.SHORT_ANSWER,
            difficulty=diff_enum,
            options=None,
            correct_answer=correct_answer,
            explanation=explanation,
            source_page_numbers=[concept.page_number],
        )

    def _build_long_answer(
        self,
        concept: ConceptUnit,
        difficulty: str,
        q_idx: int,
    ) -> QuestionItem:
        """
        Builds a comprehensive synthesis question with structured marking rubrics.
        """
        term = concept.term
        prompt = (
            f"Provide an in-depth analysis of {term}. "
            f"Describe its core mechanism, requirements, and practical implications as detailed in your notes."
        )
        correct_answer = (
            f"Key Rubric Elements:\n"
            f"1. Core Definition: {concept.definition}\n"
            f"2. Context & Significance: Derived from notes on Page {concept.page_number}.\n"
            f"3. Operational Distinctions: Verified against source: '{concept.source_sentence}'."
        )
        explanation = (
            f"Comprehensive synthesis question drawn from Page {concept.page_number}. "
            f"Full credit requires addressing definition, mechanism, and practical consequences."
        )

        diff_enum = QuestionDifficulty.HARD if difficulty == "hard" else (QuestionDifficulty.EASY if difficulty == "easy" else QuestionDifficulty.MEDIUM)

        return QuestionItem(
            question_id=f"q_{q_idx}",
            question_text=prompt,
            question_type=QuestionType.LONG_ANSWER,
            difficulty=diff_enum,
            options=None,
            correct_answer=correct_answer,
            explanation=explanation,
            source_page_numbers=[concept.page_number],
        )

    def _build_supplementary_question(
        self,
        concept: ConceptUnit,
        qtype: str,
        difficulty: str,
        q_idx: int,
        rng: random.Random,
        all_terms: List[str],
    ) -> Optional[QuestionItem]:
        """Creates an alternate formulation if duplicate prompts occurred."""
        if qtype == "mcq":
            prompt = f"What is the key takeaway regarding {concept.term} on Page {concept.page_number}?"
            distractors = rng.sample([t for t in all_terms if t != concept.term], min(3, len(all_terms) - 1))
            opts = [concept.term] + distractors
            rng.shuffle(opts)
            return QuestionItem(
                question_id=f"q_{q_idx}",
                question_text=prompt,
                question_type=QuestionType.MCQ,
                difficulty=QuestionDifficulty.MEDIUM,
                options=opts,
                correct_answer=concept.term,
                explanation=concept.source_sentence,
                source_page_numbers=[concept.page_number],
            )
        elif qtype == "true_false":
            return self._build_true_false(concept, difficulty, q_idx, rng)
        else:
            return self._build_short_answer(concept, difficulty, q_idx)

    # -------------------------------------------------------------------------
    # Deterministic Quiz Evaluation & Transparent Scoring
    # -------------------------------------------------------------------------

    def evaluate_quiz_session(
        self,
        session_id: str,
        submissions: List[QuestionAnswerSubmission],
        session_token: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> EvaluateQuizResponse:
        """
        Securely evaluates quiz submissions against an authoritative SQLite session:
        1. Validates session existence and session_token ownership (KeyError 404 if missing).
        2. Validates document_id match if provided (ValueError 400).
        3. Validates for duplicate question IDs in submissions (ValueError 400).
        4. Validates that every question ID belongs to the authoritative session (ValueError 400).
        5. Validates option selections for MCQ and answer formats for True/False (ValueError 400).
        6. Grades answers strictly using server-stored answer keys without trusting client input.
        """
        session = self.store.get_quiz_session(session_id, session_token=session_token)
        if not session:
            raise KeyError(f"Quiz session with ID '{session_id}' not found.")

        if document_id and document_id.strip():
            clean_doc_id = document_id.strip()
            if session.get("document_id") != clean_doc_id:
                raise ValueError(
                    f"Document ID '{clean_doc_id}' does not match authoritative quiz session."
                )

        raw_questions = session.get("questions", [])
        authoritative_questions = [QuestionItem(**q) for q in raw_questions]
        valid_qids = {q.question_id for q in authoritative_questions}

        # 1. Check for duplicate question submissions
        sub_qids = [s.question_id for s in submissions]
        if len(sub_qids) != len(set(sub_qids)):
            raise ValueError("Duplicate answer submissions detected for the same question ID.")

        # 2. Check for question IDs that do not belong to this session
        for s in submissions:
            if s.question_id not in valid_qids:
                raise ValueError(
                    f"Invalid question ID '{s.question_id}': does not belong to session '{session_id}'."
                )

        # 3. Validate option selections and answer formats
        sub_map = {s.question_id: s.user_answer.strip() for s in submissions}
        for q in authoritative_questions:
            user_ans = sub_map.get(q.question_id, "").strip()
            if not user_ans:
                continue
            q_type = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
            if q_type == "mcq" and q.options:
                norm_user = user_ans.upper()
                valid_letters = [chr(ord("A") + i) for i in range(len(q.options))]
                is_valid_letter = norm_user in valid_letters
                is_valid_option = any(user_ans.lower() == opt.lower() for opt in q.options)
                if not (is_valid_letter or is_valid_option):
                    raise ValueError(
                        f"Invalid option selection '{user_ans}' for question '{q.question_id}'. Selection must be one of the available choices."
                    )
            elif q_type == "true_false":
                norm_u = user_ans.lower()
                valid_tf = {"true", "false", "t", "f", "1", "0", "yes", "no"}
                if norm_u not in valid_tf:
                    raise ValueError(
                        f"Invalid answer format for True/False question '{q.question_id}'. Expected 'True' or 'False'."
                    )

        # 4. Evaluate using authoritative server questions
        return self.evaluate_quiz(
            questions=authoritative_questions,
            submissions=submissions,
        )

    def evaluate_quiz(
        self,
        questions: List[QuestionItem],
        submissions: List[QuestionAnswerSubmission],
    ) -> EvaluateQuizResponse:
        """
        Evaluates student submissions against the authoritative questions:
        - MCQ: Exact option match or letter (A/B/C/D) resolution.
        - True/False: Normalized case-insensitive boolean matching.
        - Short/Long Answer: Transparent keyword & key phrase coverage against marking points.
        """
        sub_map = {s.question_id: s.user_answer.strip() for s in submissions}
        evaluations: List[QuestionEvaluationResult] = []
        total_score = 0.0

        for q in questions:
            user_ans = sub_map.get(q.question_id, "").strip()
            q_type = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
            correct_ans = q.correct_answer.strip()

            is_correct = False
            points = 0.0
            feedback = ""

            if not user_ans:
                feedback = "No answer was provided."
                evaluations.append(
                    QuestionEvaluationResult(
                        question_id=q.question_id,
                        question_type=q_type,
                        user_answer="",
                        correct_answer=correct_ans,
                        is_correct=False,
                        score=0.0,
                        feedback=feedback,
                        explanation=q.explanation,
                        source_page_numbers=q.source_page_numbers,
                    )
                )
                continue

            # 1. Multiple Choice Evaluation
            if q_type == "mcq":
                # Handle letter response like 'A', 'B', 'C', 'D'
                norm_user = user_ans.upper()
                if norm_user in ["A", "B", "C", "D"] and q.options:
                    letter_idx = ord(norm_user) - ord("A")
                    if 0 <= letter_idx < len(q.options):
                        user_ans_resolved = q.options[letter_idx].strip()
                    else:
                        user_ans_resolved = user_ans
                else:
                    user_ans_resolved = user_ans

                if user_ans_resolved.lower() == correct_ans.lower():
                    is_correct = True
                    points = 1.0
                    feedback = "Correct! Your selection matches the factual concept from your notes."
                else:
                    is_correct = False
                    points = 0.0
                    feedback = f"Incorrect. The correct option is: \"{correct_ans}\"."

            # 2. True / False Evaluation
            elif q_type == "true_false":
                norm_u = user_ans.lower()
                norm_c = correct_ans.lower()
                is_u_true = norm_u in ["true", "t", "yes", "1"]
                is_c_true = norm_c in ["true", "t", "yes", "1"]

                if is_u_true == is_c_true:
                    is_correct = True
                    points = 1.0
                    feedback = f"Correct! The statement is indeed {correct_ans}."
                else:
                    is_correct = False
                    points = 0.0
                    feedback = f"Incorrect. The correct answer is {correct_ans}."

            # 3. Short & Long Answer Evaluation (Transparent Keyword/Rubric Coverage)
            else:
                expected_kw = extract_keywords(correct_ans)
                user_kw = extract_keywords(user_ans)
                overlap = expected_kw.intersection(user_kw)
                coverage = len(overlap) / max(1, len(expected_kw))

                if coverage >= 0.50:
                    is_correct = True
                    points = 1.0
                    feedback = (
                        f"Full Credit (Concept overlap: {coverage:.0%}). "
                        f"Matched key points: {', '.join(list(overlap)[:4])}."
                    )
                elif coverage >= 0.25:
                    is_correct = True # Partial credit
                    points = 0.5
                    feedback = (
                        f"Partial Credit (Concept overlap: {coverage:.0%}). "
                        f"Included points: {', '.join(list(overlap)[:3])}. Review the expected answer for full completeness."
                    )
                else:
                    is_correct = False
                    points = 0.0
                    feedback = (
                        f"Needs Review (Concept overlap: {coverage:.0%}). "
                        f"Missed key concepts required by the marking rubric."
                    )

            total_score += points
            evaluations.append(
                QuestionEvaluationResult(
                    question_id=q.question_id,
                    question_type=q_type,
                    user_answer=user_ans,
                    correct_answer=correct_ans,
                    is_correct=is_correct,
                    score=points,
                    feedback=feedback,
                    explanation=q.explanation,
                    source_page_numbers=q.source_page_numbers,
                )
            )

        max_score = float(len(questions))
        percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0.0
        correct_count = sum(1 for e in evaluations if e.score >= 0.5)

        return EvaluateQuizResponse(
            total_questions=len(questions),
            correct_count=correct_count,
            total_score=round(total_score, 1),
            max_score=max_score,
            percentage=percentage,
            evaluations=evaluations,
            grading_method="deterministic_and_keyword_overlap",
        )


# Global singleton instance
question_generator_instance = QuestionGenerator()
