# Contributing to Actuarial AI Case Studies

Thank you for your interest in contributing to the **Actuarial AI Case Studies** repository! We welcome contributions from actuaries, data scientists, AI practitioners, and anyone interested in showcasing how Artificial Intelligence can be applied to actuarial work.

## How to Contribute

### 1. Familiarize Yourself with the Repository

- **Review the README:** This repository curates real-world case studies that highlight the use of AI in actuarial science.
- **Templates:** Before you start, please review the [templates](./templates/) provided. They are designed to help structure your submission and ensure consistency across contributions.

### 2. Choosing Your Contribution

You can contribute in two main ways:
- **Case Studies:** Submit detailed case studies that explore various applications of AI in areas like risk modeling, pricing, forecasting, claims analysis, and more.
- **Templates:** Suggest updates or improvements to the case study templates to help future contributors. If you have a new idea for structuring content, please propose it via an issue or direct contact.

### 3. Making a Submission

#### i. As a Member of the Broader Actuarial Community

- **Fork & Clone:** Fork this repository and clone it to your local machine.
- **Create a Branch:** Create a new branch off `dev` for your contribution. We recommend using a descriptive/unique branch name (e.g. `case_study_wilson` for adding a case study written from some author Wilson).
- **Add Your Contribution:** 
  - Place your case study in the `case-studies/` folder. Ensure your submission follows the guidelines and the template provided.
  - If submitting template improvements or suggestions, add your changes or new files in the `templates/` folder.
- **Testing Your Contribution:** Review your changes locally to ensure they display correctly and that all content aligns with our style guidelines.

#### ii. As a Member of the Workstream Core Group

You can fork a repo and submit new work via Pull Request as above, but the following workflow should be simpler as it allows you to add to the official code base directly rather than forking & opening PRs.

##### a. Directly in GitHub GUI

- **Branch from `dev`:** Create a new branch from the GUI by
  - Navigating from the branch dropdown > "View all branches" > (green button) "New Branch"
  - Choose `dev` as the source and assign a descriptive branch name
  - Click "Create new branch"
- **Ensure you are on the new branch:** After creating the new branch you should see it in the list of all branches. Click on it to return to the home page/README for the new branch. The content will be identical to the version on `dev`. You can also switch between branches from this page by using the branch selector drop down (top left) - ensure that it shows you are currently viewing your new branch.
- **Add Your Contribution:** 
  - Navigate the file directory by clicking on any sub-directory
  - Files can be uploaded through the web GUI from the top right: Add file > Upload files
  - Place your case study in the `case-studies/` folder. Ensure your submission follows the guidelines and the template provided
  - You can also edit the README.md to add your case study to the list:
    - Click on README.md
    - At the top right you will see a pencil icon - Click this or the drop down to "Edit in place"
    - Add the relevant info following the same syntax you see for other list items - entries in the table are pipe ("|") -delimited; hyperlinks use the syntax `[Text to display](url)`, etc.
    - This is known as [Markdown syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) with plenty of content available online if you need
  - When done adding or editing content, changes must be committed.
    - Top right of editor > "Commit changes"
    - Keep the commit message short yet descriptive. The default commit message may be "Update README.md" - you might add "Add case study by Wilson et. al. to table" in the extended description.
    - "Commit directly to the [branch_name]" is correct assuming you have first created a new branch off dev; if not, choosing the second option would both create the new branch now and add this commit to it.
    - "Commit Changes" (green button) to finalize the commit
  - If submitting template improvements or suggestions, add your changes or new files in the `templates/` folder
- **Testing Your Contribution:** Review your changes locally (or on the new branch) to ensure they display correctly and that all content aligns with our style guidelines

##### b. Command Line

### 4. Submit a Pull Request

- **Write a Clear Description:** In your pull request, provide a brief explanation of your changes and the rationale behind them.
- **Reference Issues:** If your contribution addresses an open issue, please reference it in your pull request comments.
- **Review Process:** Once submitted, a maintainer will review your contribution. They may request changes or provide feedback to ensure the submission fits with the repository's goals.

### 5. Merge your branch back into `dev`

- After opening the pull request, you will likely see there are no merge conflicts to resolve. This means nobody has added commits onto the `dev` branch since you merged your feature branch off it.
- Click the green button "Merge pull request"
- Choose "Squash and merge" - this will reduce possible multiple commits added on your feature branch down to one commit merged into the `dev` branch
- Your commit message will summarize the changes made over any new commits on the feature branch. For example, if you "Added case study document", "Updated README", "Fixed bug found in one/of/the/files.R", and "Minor grammar edits" over 4 separate commits, we do not need to preserve this full commit history in the log for dev. Your single squash-merge commit message will more concisely log that you "Added Wilson Case Study with README updates".

## Guidelines for Submissions

- **Clarity & Accuracy:** Ensure all case studies are clear, well-documented, and accurate. Provide data sources and references when applicable.
- **Reproducibility:** Where possible, include code snippets, detailed methodologies, and steps to reproduce the results presented in your case study.
- **Formatting:** Follow the Markdown style and structure outlined in the provided templates to maintain consistency.
- **Licensing:** By contributing, you agree that your submissions will be made available under the [MIT License](./LICENSE).

## Additional Help

If you have questions or need assistance with your contribution, feel free to:
- **Open an Issue:** Use the GitHub issue tracker to ask questions or seek guidance.
- **Contact Directly:** You can also reach out via [email](mailto:simon.hatzesberger@gmail.com).

Thank you for your contribution and for helping us advance the application of AI in actuarial science!

---

*This document may be updated as needed. Please refer to it regularly to ensure compliance with the latest guidelines.*
