# Contributing to Actuarial AI Case Studies

Thank you for your interest in contributing! This guide walks you through the entire process step by step. **No prior GitHub experience is required** — just follow the instructions below.

> [!NOTE]
> **What is GitHub?** GitHub is a platform for collaboratively managing and sharing files — similar to a shared drive, but with built-in version control. Every change is tracked, so nothing is ever lost. This repository uses GitHub to collect and publish actuarial AI case studies.

---

## Prerequisites

Before you begin, make sure you have:

1. **A GitHub account** — Sign up for free at [github.com](https://github.com/signup) if you do not have one yet.

That is all you need. The workflow below uses **forking**, which means you create your own copy of the repository under your GitHub account. You do not need any special permissions.

> [!IMPORTANT]
> You must be **logged in** to your GitHub account to fork, create branches, and edit files. If you do not see the **"Fork"** button or editing options, you are not logged in.

---

## What You Can Contribute

- **Catalog entries** — Add references to published papers, articles, or code repositories to the [Case Study Catalog](./case-studies/). This is the most common contribution and is covered in detail in **Part A** below.
- **Full case studies** — Submit a complete, self-contained case study with a Jupyter notebook, data, and documentation. See **Part B** below.
- **Templates** — Propose improvements to the [case study templates](./templates/). Open an [issue](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/issues) or [contact us](mailto:simon.hatzesberger@gmail.com) with your suggestion.

---

## Part A — Adding a Case Study Entry to the Catalog

This is the most common workflow. You will add an entry to the catalog file (`case-studies/README.md`) that describes a case study and links to its resources (article, code, dataset, etc.).

### Step 1: Fork the Repository

A **fork** is your personal copy of the repository. You make all your changes there, then propose them back to the original repository via a pull request.

1. **Navigate** to the repository: [github.com/IAA-AITF/Actuarial-AI-Case-Studies](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies)
2. **Confirm you are logged in** — your profile picture should be visible in the top-right corner. If you see a **"Sign in"** button instead, log in first.
3. **Click** the **"Fork"** button (top-right of the page).
4. On the fork creation page, leave the defaults and **click** **"Create fork"**.
5. GitHub will redirect you to your fork. You can confirm this by checking the repository name at the top — it should say **`your-username/Actuarial-AI-Case-Studies`** with a note "forked from IAA-AITF/Actuarial-AI-Case-Studies".

> [!TIP]
> You only need to fork once. If you have already forked this repository, navigate to your fork at `github.com/your-username/Actuarial-AI-Case-Studies` instead.

### Step 2: Create Your Working Branch

You will create a branch on your fork to keep your changes organized. This is your private workspace — you can make changes freely without affecting anyone else.

> [!NOTE]
> **What is a branch?** Think of a branch as your own private workspace. You can make changes freely without affecting anyone else's work. Once you are done, you merge your changes back via a pull request.

1. On your fork's main page, **click** the branch dropdown (top-left of the page, it likely says **"main"** or **"dev"**).
2. **Type** a short, descriptive branch name with no spaces (e.g., `entries_wilson`, `catalog_smith`).
3. **Click** **"Create branch: your-branch-name from 'dev'"** (or from `main` if `dev` is not available).

> [!IMPORTANT]
> If a **`dev`** branch exists, create your branch from **`dev`** — it contains the latest working version. If only `main` is available, use `main`.

### Step 3: Switch to Your Branch

1. **Click** the branch dropdown (top-left).
2. **Select** your newly created branch from the list.
3. **Verify** that the branch dropdown now displays your branch name.

> [!WARNING]
> **Always confirm you are on your own branch before making any edits.** If the branch dropdown shows `main` or `dev`, switch to your branch first.

### Step 4: Open the Catalog File for Editing

1. **Click** the **`case-studies/`** folder in the file listing.
2. **Click** the **`README.md`** file inside that folder. This is the catalog file — it contains all case study entries, organized by year.
3. **Click** the **pencil icon** (top-right of the file content area) to enter edit mode. You can also click the dropdown arrow next to the pencil and select **"Edit in place"**.

You are now in the file editor. You will see two tabs at the top: **Edit** and **Preview**.

> [!NOTE]
> **What is this file?** The `case-studies/README.md` is the catalog of all case studies in this repository. Each entry describes a case study and links to its resources. Entries are grouped by year, with the most recent year at the top.

### Step 5: Add Your Entry

1. **Scroll** to the correct year section in the file (e.g., the `## 2026` section). Entries within a year should be ordered by date (most recent first).
2. **Copy** the template below and **paste** it at the appropriate position in the file.
3. **Fill in** each field with the details of your case study.

**Entry template** — copy and paste this, then replace the placeholder values:

````markdown
### Title of the Case Study
- **Author:** Author Name(s)
- **Date:** YYYY-MM-DD
- **Resources:** [Article](https://link-to-article), [Code](https://link-to-code)
- **Type:** Case Study
- **Level:** 🟩⬜⬜ Beginner
- **Field:** P&C
- **Primary Topics:** `Topic 1`, `Topic 2`
- **Secondary Topics:** `Topic 1`, `Topic 2`
- **Language(s):** English
- **Programming Language(s):** Python
- **Methods and/or Models:** Brief description of the methods used.
- **Notes:** Additional context, or – if none.
- **Abstract/Summary:**
    Paste the abstract or a brief summary here.
<br>
````

> [!TIP]
> **How to format links:** Use the syntax `[Display Text](https://url)`. For example: `[Article (arXiv)](https://arxiv.org/abs/1234.56789)`. Multiple resources are separated by commas.

> [!TIP]
> **Check your formatting** by switching to the **Preview** tab at the top of the editor. This shows you exactly how the entry will look on the published page. Switch back to **Edit** to continue making changes.

**Field reference:**

| Field | Accepted values |
|:------|:----------------|
| **Type** | `Case Study`, `Tutorial`, `White Paper`, `Educational` |
| **Level** | `🟩⬜⬜ Beginner`, `🟨🟨⬜ Advanced`, `🟥🟥🟥 Expert` |
| **Field** | `Life`, `P&C`, `Health`, `General` |
| **Date** | ISO 8601 format: `YYYY-MM-DD` (e.g., `2025-06-22`) |
| **Programming Language(s)** | `Python`, `R`, or `–` if not applicable |

> [!TIP]
> **When in doubt, look at existing entries** in the same file. They show the exact syntax and formatting for every field.

### Step 6: Save Your Work (Commit)

Once you have added or edited your entry:

1. **Click** the green **"Commit changes..."** button (top-right of the editor).
2. A dialog will appear:
   - **Commit message** — Write a short description of what you changed (e.g., "Add Smith et al. reinforcement learning entry").
   - **Extended description** — Optional. You can add more detail if needed.
   - **Commit directly to `[your-branch-name]`** — This option should already be selected. Confirm it shows your branch name.
3. **Click** the green **"Commit changes"** button in the dialog.

> [!NOTE]
> **What is a commit?** A commit is like saving your work. Each commit creates a snapshot of your changes. You can make multiple commits — for example, one for each entry you add.

> [!IMPORTANT]
> Make sure the dialog says **"Commit directly to `[your-branch-name]`"**, not to `dev` or `main`. If it shows the wrong branch, cancel and switch to your branch first (see Step 3).

> [!TIP]
> **Save often.** You can commit after each entry you add. If something goes wrong, you can always go back to a previous commit.

### Step 7: Review Your Changes

1. After committing, you are returned to the file view. **Click** the **Preview** view to review how your entries will appear.
2. **Check** that all links work by clicking them.
3. **Verify** that the formatting looks correct — bold text, bullet points, and headings should all render properly.

If you spot errors, repeat Steps 4–6 to make corrections.

### Step 8: Create a Pull Request

When you are satisfied with your changes, you need to propose merging them into the original repository. This is done through a **pull request**.

> [!NOTE]
> **What is a pull request?** A pull request (PR) is a formal proposal to merge your changes into the original repository. It lets maintainers review what you changed before the changes go live.

1. **Navigate** to your fork's main page on GitHub.
2. GitHub will show a banner saying your branch is ahead of the original repository, with a **"Contribute"** button or a **"Compare & pull request"** button. **Click** it.
   - If you do not see this banner: **click** the **"Pull requests"** tab on the **original** repository, then **click** the green **"New pull request"** button. Click **"compare across forks"**, select the original repository's `dev` branch as the **base**, and your fork's branch as the **compare** branch.
3. On the pull request form:
   - **Title** — Write a short title (e.g., "Add 3 new case study entries").
   - **Description** — Briefly describe what you added or changed.
   - **Base branch** — Confirm it points to the original repository's **`dev`** branch (or `main` if `dev` does not exist).
4. **Click** the green **"Create pull request"** button.

### Step 9: Wait for Review

After creating the pull request, a maintainer will review your contribution. You may receive feedback or change requests — GitHub will notify you by email. Once approved, a maintainer will merge your changes.

> [!TIP]
> You can check the status of your pull request at any time by navigating to the **"Pull requests"** tab on the original repository.

### Step 10: Verify Your Contribution

After your pull request has been merged:

1. **Navigate** to the original repository: [github.com/IAA-AITF/Actuarial-AI-Case-Studies](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies)
2. **Switch** to the **`dev`** branch using the branch dropdown.
3. **Open** the `case-studies/` folder and click `README.md`.
4. **Confirm** that your entries appear in the catalog.

Your contribution is now part of the shared `dev` branch. A maintainer will periodically transfer approved changes from `dev` to `main`.

---

## Part B — Submitting a Full Case Study (with Notebook and Files)

If you are contributing a complete, runnable case study (not just a catalog entry), follow the same fork-and-branch workflow as Part A, with these additional steps:

### Preparing Your Files

Your case study directory should include:

| File | Description |
|:-----|:------------|
| `your_case_study.ipynb` | Jupyter notebook with code, narrative, and visualizations |
| `your_case_study.html` | Rendered HTML export of the notebook |
| `requirements.txt` | Python dependencies with version numbers |
| `README.md` | Description, getting started instructions, and key takeaways |
| Any additional data files | Datasets, models, or configuration files |

> [!TIP]
> Use the provided [templates](./templates/) as a starting point. They include the recommended structure and formatting for both Jupyter Notebook and RMarkdown case studies.

### Uploading Your Files

1. **Fork and create a branch** (see Part A, Steps 1–3).
2. **Navigate** to the `case-studies/` folder, then into the appropriate year folder (e.g., `2025/`).
3. **Click** **"Add file"** (top-right) and then **"Upload files"**.
4. **Drag and drop** your case study folder or select files from your computer.
5. **Commit** the uploaded files to your branch (see Part A, Step 6).
6. **Add a catalog entry** to `case-studies/README.md` following the instructions in Part A, Steps 4–6.
7. **Create a pull request** (Part A, Step 8) and wait for a maintainer to review and merge it.

---

## For Experienced Git Users: CLI Workflow

If you prefer working locally with Git, here is the quick-reference workflow:

```bash
# 1. Fork the repository on GitHub (use the "Fork" button), then clone your fork
git clone https://github.com/your-username/Actuarial-AI-Case-Studies.git
cd Actuarial-AI-Case-Studies

# 2. Create a branch from dev (or main if dev does not exist)
git checkout dev
git checkout -b your-branch-name

# 3. Make your changes (edit case-studies/README.md, add files, etc.)

# 4. Stage and commit
git add .
git commit -m "Add your-case-study entry"

# 5. Push to your fork
git push origin your-branch-name

# 6. Open a pull request on GitHub
#    Go to the original repository and click "Compare & pull request",
#    or use:  gh pr create --base dev --repo IAA-AITF/Actuarial-AI-Case-Studies
```

> [!TIP]
> If your fork is behind the original repository, sync it before starting: on your fork's GitHub page, click **"Sync fork"**, or run `git fetch upstream && git merge upstream/dev`.

---

## Quick Reference: Markdown Syntax

The catalog file uses [Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax), a simple formatting language. Here are the patterns you will need:

| What you type | What it produces |
|:--------------|:-----------------|
| `**bold text**` | **bold text** |
| `` `code` `` | `code` |
| `[Link Text](https://url)` | [Link Text](https://url) |
| `### Heading` | A heading (level 3) |
| `- List item` | A bullet point |

---

## Guidelines for Submissions

- **Clarity & Accuracy** — Ensure case studies are well-documented with data sources and references where applicable.
- **Reproducibility** — Include code, detailed methodologies, and steps to reproduce results.
- **Formatting** — Follow the Markdown structure outlined in the provided [templates](./templates/) and existing catalog entries.
- **Licensing** — By contributing, you agree that your submissions will be available under the [MIT License](./LICENSE) (code) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (content).

---

## Need Help?

- [Open an issue](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/issues) on the GitHub issue tracker
- [Contact us via email](mailto:simon.hatzesberger@gmail.com)

---

*This document may be updated. Please refer to it before each submission.*
