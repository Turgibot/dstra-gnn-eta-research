# SN Computer Science Submission Checklist

Based on: https://link.springer.com/journal/42979/submission-guidelines

## 📋 Pre-Submission Requirements

### Manuscript Submission
- [ ] Work has not been published before
- [ ] Work is not under consideration for publication elsewhere
- [ ] Publication has been approved by all co-authors
- [ ] Publication has been approved by responsible authorities at the institute
- [ ] All relevant editable source files are provided (.tex or .docx)
- [ ] Complete set of editable source files submitted (not just PDF)

### Permissions
- [ ] Permission obtained for any figures/tables/text from previously published works
- [ ] Evidence of permission included for both print and online format
- [ ] All reused material properly attributed

---

## 📄 Title Page Requirements

### Title
- [ ] Title is concise and informative
- [ ] Short title provided (for running head)

### Author Information
- [ ] Name(s) of all author(s) provided
- [ ] Affiliation(s) of all author(s) provided (institution, department, city, state, country)
- [ ] Clear indication of corresponding author (marked with asterisk)
- [ ] Active email address provided for corresponding author
- [ ] ORCID ID provided for all authors (16-digit format) - if available
- [ ] Address information included with affiliations (will be published)

### Abstract
- [ ] Structured abstract provided (150-250 words)
- [ ] Abstract divided into required sections:
  - [ ] **Purpose** (stating main purposes and research question)
  - [ ] **Methods**
  - [ ] **Results**
  - [ ] **Conclusion**
- [ ] Abstract word count: 150-250 words ✓ (Current: ~200 words)

### Keywords
- [ ] 4-6 keywords provided for indexing
- [ ] Keywords are relevant and appropriate ✓ (Current: 5 keywords)

### Statements and Declarations
- [ ] "Statements and Declarations" section included
- [ ] **Competing Interests** statement included
- [ ] All required declarations present (see detailed checklist below)

---

## 📝 Text Formatting

### General Formatting
- [ ] Normal, plain font used (e.g., 10-point Times Roman)
- [ ] Italics used for emphasis (not underlining)
- [ ] Automatic page numbering enabled
- [ ] No field functions used
- [ ] Tab stops used for indents (not space bar)
- [ ] Tables created using table function (not spreadsheets)
- [ ] Equations created using equation editor or MathType
- [ ] File saved in appropriate format (.tex for LaTeX, .docx for Word)

### LaTeX-Specific
- [ ] Using Springer Nature LaTeX template ✓
- [ ] All packages properly loaded
- [ ] No custom commands that conflict with template

### Final LaTeX Consolidation (do this LAST, right before upload — verified against `Template/springer-nature-template/sn-article-template/user-manual.pdf`)
- [ ] Combine `main.tex` + all `sections/*.tex` into one single `.tex` file — the template explicitly forbids `\input{...}` for submission ("Submit your LaTeX manuscript as one .tex document")
- [ ] Inline or pre-render the 8 `\input{figures/*.tikz}` calls in `methodology.tex` (`fig_dynamic_graph`, `fig_encoder`, `fig_route_encoder`, `fig_router_moe`, `fig_temporal_moe_eta`, `fig_temporal_gru_detail`, + 2 more) — same `\input` rule applies to these as to section files. Either paste each `tikzpicture` body inline, or render to PDF/PNG and switch to `\includegraphics`
- [ ] Move all figure files out of the `figures/` subfolder to sit flat next to the consolidated `.tex` file, and update every `\includegraphics`/`\input` path — manual states the submission system cannot resolve subfolder paths
- [ ] Decide whether to keep `\subfloat` (subfig package, used for the 3-panel architecture figure in methodology) or split it into separate full-width figures — manual recommends avoiding subfigures
- [ ] Drop the unused `\usepackage{adjustbox}` import (loaded, never used)
- [ ] Optional: tighten `\tabcolsep`/font in the related-work comparison table (`related_work.tex` Table 1) to clear a 2.7pt overfull-hbox warning (cosmetic, invisible at print res)
- [ ] Optional: trim abstract from ~252 words to ≤250 to match the target stated above
- [ ] Recompile the consolidated single file clean (zero warnings) and diff visually against the current modular build before upload

### Headings
- [ ] No more than three levels of displayed headings used
- [ ] Heading hierarchy is logical and consistent

### Abbreviations
- [ ] All abbreviations defined at first mention
- [ ] Abbreviations used consistently thereafter

### Footnotes
- [ ] Footnotes used for additional information (not just citations)
- [ ] Footnotes numbered consecutively
- [ ] Table footnotes use superscript lower-case letters (or asterisks for statistics)
- [ ] No footnotes to title or authors
- [ ] Footnotes used instead of endnotes

### Acknowledgments
- [ ] Acknowledgments in separate section
- [ ] Names of funding organizations written in full
- [ ] Grant or contribution numbers included if applicable ✓

---

## 📚 References

### Citation Format
- [ ] Citations identified by numbers in square brackets (e.g., [3], [1-3, 7])
- [ ] All citations in text match reference list
- [ ] No citations to unpublished works (only mentioned in text if needed)

### Reference List
- [ ] Only includes works cited in text
- [ ] Only includes published or accepted-for-publication works
- [ ] Entries numbered consecutively
- [ ] DOIs included as full DOI links where available (e.g., "https://doi.org/abc")
- [ ] Reference format follows journal style (sn-basic.bst) ✓

### Reference Types Check
- [ ] Journal articles properly formatted
- [ ] Books properly formatted
- [ ] Book chapters properly formatted
- [ ] Online documents properly formatted
- [ ] Conference proceedings properly formatted
- [ ] All DOIs are full links (not just numbers)

---

## 📊 Tables

### Table Formatting
- [ ] Tables created using table function (not spreadsheets)
- [ ] Tables numbered consecutively
- [ ] Each table has a caption
- [ ] Table captions are descriptive
- [ ] Tables are referenced in text
- [ ] Table footnotes use superscript letters (or asterisks)
- [ ] Tables are editable (not images)

### Table Content
- [ ] All tables are necessary and add value
- [ ] Tables are clear and readable
- [ ] Units are specified
- [ ] Statistical significance indicated where appropriate

---

## 🎨 Artwork and Illustrations

### Figure Requirements
- [ ] All figures are high quality and clear
- [ ] Figures numbered consecutively
- [ ] Each figure has a caption
- [ ] Figure captions are descriptive
- [ ] Figures are referenced in text
- [ ] Figures are in acceptable formats (PNG, PDF, EPS, etc.) ✓
- [ ] Figures are editable or high-resolution

### Figure Quality
- [ ] Resolution is sufficient for publication
- [ ] Text in figures is readable
- [ ] Colors are appropriate (consider colorblind readers)
- [ ] Figures are properly sized

### Figure Files
- [ ] All figure files are included
- [ ] Figure files are named appropriately
- [ ] Figures are not embedded in text (submitted separately)

---

## 📎 Supplementary Information (SI)

### If Applicable
- [ ] Supplementary Information clearly marked
- [ ] SI is necessary and adds value
- [ ] SI is properly formatted
- [ ] SI is referenced in main text

---

## 👥 Authorship Requirements

### Author Information
- [ ] All authors meet authorship criteria:
  - [ ] Made substantial contributions to conception/design OR acquisition/analysis/interpretation of data OR creation of new software
  - [ ] Drafted the work or revised it critically for important intellectual content
  - [ ] Approved the version to be published
  - [ ] Agree to be accountable for all aspects of the work

### Corresponding Author
- [ ] One author clearly designated as Corresponding Author ✓ (Nadav Voloch)
- [ ] Corresponding Author email is active and correct ✓
- [ ] Corresponding Author has approved:
  - [ ] All listed authors have approved the manuscript
  - [ ] Author names and order are correct
  - [ ] Will manage all communication with journal
  - [ ] Will provide transparency on re-use of material
  - [ ] Will ensure all disclosures/declarations are included

### Author Contributions
- [ ] Author contribution statement included ✓
- [ ] Statement specifies contribution of every author ✓
- [ ] Statement is clear and detailed

### Author Identification
- [ ] ORCID IDs provided for all authors (if available)
  - [ ] Guy Tordjman: ✓ (0009-0009-8812-0122)
  - [ ] Nadav Voloch: ⚠️ (TODO: Add ORCID ID when available)

### Affiliation
- [ ] Primary affiliation for each author is institution where majority of work was done ✓
- [ ] Current address provided if author has moved
- [ ] All affiliations are current and correct

### Author Names
- [ ] All author names are present
- [ ] All author names are correctly spelled
- [ ] Author names will appear exactly as desired in publication

---

## 📋 Statements and Declarations Section

### Required Statements
- [ ] **Competing Interests** statement included ✓
- [ ] **Funding** statement included ✓
- [ ] **Ethics approval and consent to participate** statement included ✓
- [ ] **Consent for publication** statement included ✓
- [ ] **Data availability** statement included ✓
- [ ] **Materials availability** statement included ✓
- [ ] **Code availability** statement included ✓
- [ ] **Author contribution** statement included ✓

### Competing Interests
- [ ] Financial interests disclosed (if any)
- [ ] Non-financial interests disclosed (if any)
- [ ] Statement is clear and complete ✓ ("The authors declare that they have no competing interests.")

### Funding
- [ ] All sources of funding disclosed ✓
- [ ] Grant numbers included ✓
- [ ] Funding organizations written in full ✓

### Ethics (if applicable)
- [ ] Ethics approval obtained (if research involves humans/animals)
- [ ] Ethics committee name and approval number provided
- [ ] Informed consent obtained (if applicable)
- [ ] Animal welfare statement included (if applicable)
- [ ] Current status: "Not applicable" ✓

### Data Availability
- [ ] Statement describes data availability ✓
- [ ] Statement indicates how data can be accessed ✓
- [ ] Statement complies with field standards

### Code Availability
- [ ] Statement describes code availability ✓
- [ ] Statement indicates how code can be accessed ✓
- [ ] Statement complies with field standards

---

## 🔬 Research-Specific Requirements

### Data Transparency
- [ ] All data and materials support published claims
- [ ] Data and materials comply with field standards
- [ ] Software application or custom code supports published claims
- [ ] Code complies with field standards

### Methods Section
- [ ] Methods section is complete and clear
- [ ] All experimental procedures described
- [ ] Statistical methods described
- [ ] Software/tools properly cited
- [ ] If LLM/AI tools used, properly documented in Methods section

### Reproducibility
- [ ] Methods are described in sufficient detail for reproduction
- [ ] Parameters and settings are specified
- [ ] Datasets are described

---

## 📝 Language and Style

### Language
- [ ] Manuscript is in English
- [ ] Language is clear and professional
- [ ] Grammar and spelling are correct
- [ ] Technical terms are used correctly

### Style
- [ ] Writing is clear and concise
- [ ] Passive/active voice used appropriately
- [ ] Tense is consistent
- [ ] First person used appropriately (if allowed)

---

## 🔍 Final Checks

### Before Submission
- [ ] All sections are complete
- [ ] All figures and tables are included
- [ ] All references are complete and correct
- [ ] All required statements are included
- [ ] Author information is complete and correct
- [ ] Corresponding author is clearly indicated
- [ ] All co-authors have approved the manuscript
- [ ] Manuscript follows template format
- [ ] All source files are editable
- [ ] PDF is generated correctly

### File Checklist
- [ ] Main manuscript file (.tex or .docx)
- [ ] All figure files (separate files)
- [ ] Bibliography file (.bib) if using LaTeX
- [ ] Any supplementary files
- [ ] Cover letter (if required)

### LaTeX-Specific Files
- [ ] main.tex ✓ (dev copy only — must be consolidated into one single file before upload, see "Final LaTeX Consolidation" above)
- [ ] references.bib ✓
- [ ] All figure files flattened to the same directory as the consolidated .tex file (NOT in a figures/ subfolder — submission system can't resolve subfolder paths; current dev tree keeps them in figures/, that's fine for development only)
- [ ] sn-jnl.cls class file ✓
- [ ] sn-basic.bst bibliography style ✓

---

## ⚠️ Common Mistakes to Avoid

- [ ] Using field functions
- [ ] Using space bar for indentation
- [ ] Embedding figures in text (should be separate files)
- [ ] Using spreadsheets for tables
- [ ] Missing required declarations
- [ ] Incorrect reference format
- [ ] Missing DOIs in references
- [ ] Incomplete author information
- [ ] Not indicating corresponding author clearly
- [ ] Missing ORCID IDs (if available)
- [ ] Incomplete funding information
- [ ] Missing ethics statements (if applicable)
- [ ] Submitting only PDF (must include source files)

---

## 📌 Notes

### Current Status Summary:
- ✅ **Title**: Complete and informative
- ✅ **Abstract**: Structured, 150-250 words, all sections present
- ✅ **Keywords**: 5 keywords provided
- ✅ **Authors**: Both authors listed with affiliations
- ✅ **Corresponding Author**: Nadav Voloch (marked with *)
- ✅ **ORCID**: Guy has ORCID, Nadav needs to add ORCID
- ✅ **Declarations**: All required sections present
- ✅ **Author Contributions**: Statement included
- ✅ **References**: Using sn-basic.bst style
- ✅ **Figures**: All figures in figures/ directory
- ✅ **LaTeX**: Using Springer Nature template

### Action Items:
1. ⚠️ Add ORCID ID for Nadav Voloch (currently has TODO comment)
2. ✅ Verify all figures are high quality
3. ✅ Verify all references have DOIs where available
4. ✅ Final proofread before submission

---

## 📞 Submission Process

### When Ready to Submit:
1. [ ] Complete all checklist items above
2. [ ] Review manuscript one final time
3. [ ] Ensure all co-authors have approved
4. [ ] Prepare cover letter (if required)
5. [ ] Go to: https://link.springer.com/journal/42979/submission-guidelines
6. [ ] Click "Submit manuscript"
7. [ ] Upload all required files
8. [ ] Complete submission form

---

**Last Updated**: Based on guidelines as of December 2024
**Source**: https://link.springer.com/journal/42979/submission-guidelines
