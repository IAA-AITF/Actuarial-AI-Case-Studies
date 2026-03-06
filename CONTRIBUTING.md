# Contributing to Actuarial AI Case Studies

Thank you for your interest in contributing! This guide walks you through the entire process step by step. **No prior GitHub experience is required** — just follow the instructions below.

> [!NOTE]
> **What is GitHub?** GitHub is a platform for collaboratively managing and sharing files — similar to a shared drive, but with built-in version control. Every change is tracked, so nothing is ever lost. This repository uses GitHub to collect and publish actuarial AI case studies.

---

## Prerequisites

Before you begin, make sure you have:

1. **A GitHub account** — Sign up for free at [github.com](https://github.com/signup) if you do not have one yet.
2. **Write access** to this repository — If you are part of the IAA AI Task Force, request access from a maintainer by [sending an email](mailto:simon.hatzesberger@gmail.com). Your access is tied to your GitHub account.

> [!IMPORTANT]
> You must be **logged in** to your GitHub account to create branches and edit files. If you do not see editing options (e.g., the **"New branch"** button), you are either not logged in or your access has not been set up yet.

---

## What You Can Contribute

- **Catalog entries** — Add references to published papers, articles, or code repositories to the [Case Study Catalog](./case-studies/). This is the most common contribution and is covered in detail in **Part A** below.
- **Full case studies** — Submit a complete, self-contained case study with a Jupyter notebook, data, and documentation. See **Part B** below.
- **Templates** — Propose improvements to the [case study templates](./templates/). Open an [issue](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies/issues) or [contact us](mailto:simon.hatzesberger@gmail.com) with your suggestion.

---

## Part A — Adding a Case Study Entry to the Catalog

This is the most common workflow. You will add an entry to the catalog file (`case-studies/README.md`) that describes a case study and links to its resources (article, code, dataset, etc.).

### Step 1: Open the Repository

1. **Navigate** to the repository: [github.com/IAA-AITF/Actuarial-AI-Case-Studies](https://github.com/IAA-AITF/Actuarial-AI-Case-Studies)
2. **Confirm you are logged in** — your profile picture should be visible in the top-right corner of the page. If you see a **"Sign in"** button instead, log in first.
3. You will see the repository's main page with a file listing and a README displayed below it. At the top-left, a dropdown shows the current branch (likely **main**).

### Step 2: Create Your Working Branch

You will create a personal copy of the `dev` branch to work on. This keeps your changes separate until they are ready to be merged.

> [!NOTE]
> **What is a branch?** Think of a branch as your own private workspace. You can make changes freely without affecting anyone else's work. Once you are done, you merge your changes back into the shared `dev` branch.

1. **Click** the branch dropdown (top-left of the page, it likely says **"main"**).
2. **Click** **"View all branches"** at the bottom of the dropdown.
3. On the Branches page, **click** the green **"New branch"** button.
4. A dialog will appear:
   - **New branch name** — Choose a short, descriptive name with no spaces. Use your name or initials so others can identify it (e.g., `entries_wilson`, `catalog_smith`).
   - **Source** — Make sure **`dev`** is selected as the source.
5. **Click** **"Create new branch"**.

> [!IMPORTANT]
> Always select **`dev`** as the source — not `main`. The `dev` branch contains the latest working version. The `main` branch is only updated by maintainers.

> [!TIP]
> You will see your new branch listed on the Branches page. You can always return to this page to find or switch between branches.

### Step 3: Switch to Your Branch

1. **Navigate** back to the repository's main page by clicking the repository name (**"Actuarial-AI-Case-Studies"**) at the top.
2. **Click** the branch dropdown (top-left). It might still show **main**.
3. **Select** your newly created branch from the list.
4. **Verify** that the branch dropdown now displays your branch name.

> [!WARNING]
> **Always confirm you are on your own branch before making any edits.** If the branch dropdown shows `main` or `dev`, you are not on your branch. Any changes made directly to `dev` or `main` will affect everyone. Switch to your branch first.

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
- **Level:** 🟩🟩⬜ Beginner
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
| **Level** | `🟩🟩⬜ Beginner`, `🟨🟨⬜ Advanced`, `🟥🟥🟥 Expert` |
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

When you are satisfied with your changes, you need to propose merging them into the shared `dev` branch. This is done through a **pull request**.

> [!NOTE]
> **What is a pull request?** A pull request (PR) is a formal proposal to merge your changes into another branch. It lets others review what you changed before the changes go live.

1. **Navigate** to the repository's main page.
2. If you recently pushed commits, GitHub will show a yellow banner at the top saying **"[your-branch] had recent pushes"** with a **"Compare & pull request"** button. **Click** it.
   - If you do not see this banner: **click** the **"Pull requests"** tab at the top of the page, then **click** the green **"New pull request"** button. Select `dev` as the **base** branch and your branch as the **compare** branch.
3. On the pull request form:
   - **Title** — Write a short title (e.g., "Add 3 new case study entries").
   - **Description** — Briefly describe what you added or changed.
   - **Base branch** — Confirm it says **`dev`** (not `main`).
4. **Click** the green **"Create pull request"** button.

### Step 9: Merge the Pull Request

After creating the pull request, you can merge it yourself if there are no conflicts.

1. On the pull request page, **scroll down**. You should see a green message: **"This branch has no conflicts with the base branch."**
2. **Click** the green **"Merge pull request"** button.
3. **Click** **"Confirm merge"**.
4. You should see a success message: **"Pull request successfully merged and closed."**

> [!WARNING]
> If you see a message about **merge conflicts**, do not force the merge. This means someone else has edited the same part of the file. Contact a maintainer for help resolving the conflict.

> [!TIP]
> After merging, GitHub may offer to **delete your branch**. You can safely delete it — the branch is no longer needed once its changes have been merged.

### Step 10: Verify Your Contribution

1. **Navigate** back to the repository's main page.
2. **Switch** to the **`dev`** branch using the branch dropdown.
3. **Open** the `case-studies/` folder and click `README.md`.
4. **Confirm** that your entries appear in the catalog.

Your contribution is now part of the shared `dev` branch. A maintainer will periodically transfer approved changes from `dev` to `main`.

---

## Part B — Submitting a Full Case Study (with Notebook and Files)

If you are contributing a complete, runnable case study (not just a catalog entry), follow the same branch workflow as Part A, with these additional steps:

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

1. **Create a branch** from `dev` (see Part A, Steps 2–3).
2. **Navigate** to the `case-studies/` folder, then into the appropriate year folder (e.g., `2025/`).
3. **Click** **"Add file"** (top-right) and then **"Upload files"**.
4. **Drag and drop** your case study folder or select files from your computer.
5. **Commit** the uploaded files to your branch (see Part A, Step 6).
6. **Add a catalog entry** to `case-studies/README.md` following the instructions in Part A, Steps 4–6.
7. **Create a pull request** and **merge** (Part A, Steps 8–9).

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
