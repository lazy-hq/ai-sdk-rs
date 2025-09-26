# Contributing

We welcome contributions! To ensure smooth collaboration, please follow these guidelines:

1. **Open an Issue First**
   Before starting work on a pull request, please open an issue to discuss your idea or bug fix.

   * This helps us confirm the change is needed, avoids duplicate work, and ensures no one else is already working on it.
   * Once the issue is agreed upon, feel free to proceed with your contribution.

2. **Report Issues**
   Use GitHub issues to report bugs, request features, or ask questions.

3. **Submit PRs**
   - Fork the repository, make your changes
   - Write change log to describe your changes in CHANGELOG.md following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.
   - Open a pull request. Please reference the related issue in your PR description.

4. **Code Style**
   Follow Rust conventions. Before committing, run:

   ```bash
   cargo fmt --all
   cargo clippy --all-features
   ```

5. **Tests**
   Add or update tests for any new features or bug fixes. Always run the full test suite with all features enabled:

   ```bash
   cargo test --all-features
   ```

6. **Commits**
   Use clear and descriptive commit messages that explain the intent of your changes.

For any questions, feel free to open an issue or continue the discussion in your pull request.
