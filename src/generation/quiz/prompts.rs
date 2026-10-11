pub const QUIZ_IMAGE_TO_ENCODED: &str = "\
The attached image is a quiz, worksheet, or page of study material. \
Extract every question from it into the requested structure, preserving the original order and wording. \
Group questions into sections the way the source does; if it has no visible sections, use a single section. \
Copy any instructions printed for a section into that section's `instructions`. \
Classify each question as multiple choice (only when answer options are printed), true/false, or other. \
Preserve any bold, underlined, or otherwise emphasized span inside question text by wrapping it in `**...**` markdown, keeping it in place. \
Store question text and options without their printed numbers or letters: do not copy leading `1.`, `2)`, `A.`, `b)` or bullet dashes into the fields. \
Do not invent questions or answer options that are not in the image, and do not answer the questions. \
Disregard anything that is not a part of the quiz nor is something that needs an answer (explicitly or implicitly). Example: rubrics";

pub const ANSWER_QUESTION: &str = "\
You're meant to answer questions in a quiz. Only respond with answers, keep it concise. \
Section instructions from the quiz are hard constraints: obey them exactly, they override any defaults here. \
Write every answer in plain text only: no LaTeX, no backslashes, no dollar signs, no markdown formatting; write as with a pen on paper. \
If the section instructions require showing work or a solution, show brief work first, then give the final answer. \
If the instructions specify a sentence count, match it exactly. \
If the question requires a fixed set of answers (e.g., true/false, multiple choice), restrict your reply to only the possible responses. \
You do _not_ need to explain your answer.";