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
- **Commit your Work (to your `feature_branch`)**
    - Changes must be committed to the git repo - this is like saving your work.
    - Top right of editor > "Commit changes"
    - Keep the commit message short yet descriptive. The default commit message may be "Update README.md" - you might add "Add case study by Wilson et. al. to table" in the extended description.
    - "Commit directly to the [`feature_branch`]" is correct assuming you have first created a new branch off dev; if not, choosing the second option would both create the new branch now and add this commit to it.
    - "Commit Changes" (green button) to finalize the commit to your `feature_branch`
- **Merge `feature_branch` to `dev`**
    - Now your feature branch will be 1 or more commits ahead of `dev` - when you have completed all work on your new feature/updates, your committed work must be merged to `dev`
    - When using the GitHUB GUI the only way to merge branches is to first open a PR and then complete the merge, closing the PR
    - On the repo home page, with your feature branch selected, click "Compare & Pull Request" or "Contribute" > "Open Pull Request"
    - Title the PR and briefly describe what is new
    - Click "Create Pull Request" (green button)
    - Select the type of Merge to perform
      - Generally, Squash & Merge will work in all cases and is preferred - this reduces multiple commits from the "source" branch (`feature_branch`) to a single commit which will be placed on the "target branch" (`dev`)
      - "Create a Merge Commit" will bring all of the commits from the source branch onto the target branch, plus an additional "Merge commit". This maintains the full history of changes committed to your feature branch on the target branch - this is almost always too verbose for the target branch log. 10 commits created in the course of developing a single new feature should be reduced to a single commit with a more concise message describing the newly added feature.
        - If you only created a single new commit on your feature branch, then the "Merge Commit" option will add 2 commits to the target branch - again, only 1 is necessary. This option is used by developer teams to preserve a full history of actions including the fact that a merge was performed. Using a Squash Merge does not preserve in history that a merge event occurred. Our workflow is designed such that every commit on the dev branch implies that a merge took place, as no work is performed directly on `dev` but instead is performed on feature branches.
  - Click the green button to execute the merge and close the PR
    - This adds a single commit to the `dev` branch
    - The new work first committed to `feature_branch` will now be visible in the file directory of the `dev` branch

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
