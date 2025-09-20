# Contributing

We welcome contributions! Here's how to get started:

## Getting Started

1. **Report Issues**: Use GitHub issues for bugs or feature requests.
2. **Submit PRs**: Fork the repo, make changes, and open a pull request.
3. **Code Style**: Follow Rust conventions. Run `cargo fmt` and `cargo clippy`.
4. **Tests**: Add tests for new features. Run `cargo test`.
5. **Commits**: Use clear, descriptive commit messages.

For questions, open an issue or discuss in PRs.

## Development Setup

### Prerequisites

- Rust 1.70+ (edition 2021)
- Cargo package manager
- Git

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/lazy-hq/ai-sdk-rs.git
cd ai-sdk-rs

# Install development dependencies
cargo install cargo-husky

# Build the project
cargo build
```

### Feature Development

The project uses feature flags to manage optional functionality:

```bash
# Build with OpenAI support
cargo build --features openai

# Build with models.dev support
cargo build --features models-dev

# Build with all features
cargo build --features full

# Build with specific features for development
cargo build --features "openai,models-dev"
```

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with specific features
cargo test --features openai
cargo test --features models-dev
cargo test --features full

# Run specific test files
cargo test --test openai_provider_integration_tests --features openai
cargo test --test models_dev_integration_tests --features models-dev

# Run tests with output
cargo test --features full -- --nocapture
```

### Test Coverage

The project includes comprehensive test coverage:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Complete workflow testing
- **Performance Tests**: Large dataset handling and concurrency
- **Error Handling Tests**: Edge cases and error scenarios

When adding new features, ensure:

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test feature integration with existing components
3. **Error Cases**: Test error handling and edge cases
4. **Documentation**: Include examples in doc tests

### Models.dev Development

When working on the models.dev feature:

#### Setup

```bash
# Enable models-dev feature for development
cargo build --features models-dev

# Run models.dev specific tests
cargo test --features models-dev

# Run models.dev examples
cargo run --example models_dev_client_example --features models-dev
cargo run --example models_dev_registry_example --features models-dev
cargo run --example models_dev_convenience_example --features models-dev
cargo run --example models_dev_complete_integration --features models-dev
```

#### Guidelines

1. **API Integration**: Test both successful and failed API calls
2. **Caching**: Test memory and disk caching functionality
3. **Concurrency**: Ensure thread-safe operations
4. **Performance**: Test with large datasets (1000+ providers, 50,000+ models)
5. **Error Handling**: Comprehensive error handling for all failure scenarios
6. **Documentation**: Add detailed inline documentation and examples

#### Adding New Convenience Functions

When adding new convenience functions:

```rust
/// Find models that support a specific capability
/// 
/// # Arguments
/// * `registry` - The provider registry to search
/// * `capability` - The capability to search for (e.g., "reasoning", "vision")
/// 
/// # Returns
/// A vector of model IDs that support the specified capability
/// 
/// # Examples
/// ```
/// use aisdk::models_dev::{ProviderRegistry, find_models_with_capability};
/// 
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let registry = ProviderRegistry::default();
/// let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
/// println!("Found {} reasoning models", reasoning_models.len());
/// # Ok(())
/// # }
/// ```
pub async fn find_models_with_capability(
    registry: &ProviderRegistry,
    capability: &str,
) -> Vec<String> {
    // Implementation
}
```

### Code Quality

#### Formatting and Linting

```bash
# Format code
cargo fmt

# Run lints
cargo clippy --features full -- -D warnings

# Run lints for specific features
cargo clippy --features models-dev -- -D warnings
cargo clippy --features openai -- -D warnings
```

#### Documentation

```bash
# Generate and test documentation
cargo doc --features full

# Check documentation examples
cargo test --doc --features full
```

### Pull Request Process

1. **Create a Branch**: Create a feature branch from `main`
2. **Make Changes**: Implement your changes with comprehensive tests
3. **Update Documentation**: Add inline documentation and update README if needed
4. **Run Checks**: Ensure all tests pass and code is formatted
5. **Submit PR**: Open a pull request with clear description

#### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass (`cargo test --features full`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy --features full -- -D warnings`)
- [ ] Documentation is updated
- [ ] Examples are tested and working
- [ ] Performance impact considered (for large changes)
- [ ] Error handling is comprehensive

### Models.dev Specific Contributions

When contributing to the models.dev feature:

1. **API Changes**: Ensure backward compatibility when changing API response handling
2. **Caching**: Consider cache invalidation strategies for new data
3. **Performance**: Profile large dataset operations
4. **Memory Usage**: Monitor memory usage with large registries
5. **Thread Safety**: Ensure all operations are thread-safe
6. **Error Messages**: Provide clear, actionable error messages

### Examples and Documentation

When adding new features:

1. **Update Examples**: Add examples demonstrating new functionality
2. **Document Public APIs**: Add comprehensive inline documentation
3. **Update README**: Update feature documentation in README.md
4. **Add Changelog**: Document breaking changes in changelog

### Performance Considerations

- **Memory Usage**: Be mindful of memory usage with large datasets
- **Concurrent Access**: Design for concurrent access where appropriate
- **Caching**: Use appropriate caching strategies
- **Error Handling**: Design error handling that doesn't impact performance

## Community

- **GitHub Discussions**: Join discussions about features and architecture
- **Issues**: Report bugs and request features
- **PR Reviews**: Participate in code reviews
- **Documentation**: Help improve documentation

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.