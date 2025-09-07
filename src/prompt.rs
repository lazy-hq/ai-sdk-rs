//! File based prompt templates that use the `tera` template engine. the module
//! searches for prompt templates in the `examples/prompts` directory relative to top level.
//! directory of the project or a directory provided by the user through an environment
//! variable. provides tools
//!     - load prompts from a directory
//!     - substitute other prompts, variables, conditionals, loops, etc
//!     - incrementally set variables a prompts with the builder pattern
//!     - logging and telemetery
//!
//! # Examples
//!
//! ```rust
//! // use ai_sdk_rs::prompt::Prompt;
//!
//! // Prompt::new("system/base")  // loads from ./prompts/system/base.prompt
//! //    .with_extension("txt") // optionally override the extension
//! //    .with("variable1", "value1")
//! //    .with("variable2", "value2")
//! //    .with("variable3", "value3")
//! //    .generate()
//! ```

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use tera::{Context, Tera};

/// Represents the environment for prompt management.
/// It contains the Tera instance for template rendering and can be configured.
#[derive(Clone)]
pub struct PromptEnvironment {
    tera: Tera,
    prompt_dir: PathBuf,
}

impl PromptEnvironment {
    /// Creates a new `PromptEnvironment` by discovering templates in the default directory,
    /// which is determined by the `PROMPT_DIR` environment variable or defaults to `./prompts`.
    pub fn new() -> Self {
        let prompt_dir = env::var("PROMPT_DIR").unwrap_or_else(|_| "examples/prompts".to_string());
        Self::from_directory(&prompt_dir)
    }

    /// Creates a new `PromptEnvironment` from a specific directory path.
    pub fn from_directory(prompt_dir_str: &str) -> Self {
        let prompt_dir = PathBuf::from(prompt_dir_str);
        let glob = format!("{prompt_dir_str}/**/*.*");
        log::debug!("Loading prompts from: {glob}");
        let mut tera = Tera::new(&glob).expect("Failed to initialize Tera");
        tera.autoescape_on(vec![]);
        Self { tera, prompt_dir }
    }
}

impl Default for PromptEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

// A type alias for a HashMap that stores prompt variables.
type PromptVariables = HashMap<String, String>;

/// A trait for objects that can be used as prompt templates.
pub trait Promptable: Sized {
    /// Renders the prompt template with the provided variables.
    fn generate(&self) -> String;
    /// Adds a variable to the prompt, if it doesn't already exist.
    fn with(self, variable: &str, value: &str) -> Self;
    /// Adds a variable to the prompt, overwriting it if it already exists.
    fn with_overwrite(self, variable: &str, value: &str) -> Self;
    /// Sets the file extension for the prompt template.
    fn with_extension(self, extension: &str) -> Self;
    /// Returns the file path of the prompt template.
    fn file_path(&self) -> PathBuf;
    /// Returns the name of the prompt.
    fn name(&self) -> &str;
    /// Returns the file extension of the prompt.
    fn extension(&self) -> &str;
    /// Returns a reference to the prompt's variables.
    fn variables(&self) -> &PromptVariables;
}

/// A struct that represents the default prompt template which is.
/// a file based prompt template.
#[derive(Clone)]
pub struct Prompt {
    // The name of the prompt, which corresponds to the template file name.
    name: String,
    // The file extension of the prompt template.
    extension: String,
    // The variables to be injected into the prompt template.
    variables: PromptVariables,
    // The environment for the prompt. holds configuration details.
    env: PromptEnvironment,
}

impl Prompt {
    /// Creates a new `Prompt` with the given name and a default extension of "prompt".
    /// It initializes a default `PromptEnvironment`.
    pub fn new(name: &str) -> Self {
        Prompt {
            name: name.to_string(),
            extension: "prompt".to_string(),
            variables: HashMap::new(),
            env: PromptEnvironment::default(),
        }
    }

    /// Creates a new `Prompt` with the given name and environment.
    pub fn new_with_env(name: &str, env: PromptEnvironment) -> Self {
        Prompt {
            name: name.to_string(),
            extension: "prompt".to_string(),
            variables: HashMap::new(),
            env,
        }
    }
}

impl Promptable for Prompt {
    /// Renders the prompt template with the provided variables.
    fn generate(&self) -> String {
        let context = Context::from_serialize(&self.variables)
            .expect("Failed to create Tera context from variables");
        let template_name = format!("{}.{}", self.name, self.extension);
        self.env
            .tera
            .render(&template_name, &context)
            .expect("Failed to render prompt")
    }

    /// Adds a variable to the prompt, if it doesn't already exist.
    /// TODO: implement different supportable types(like iterables) instead of just str
    fn with(mut self, variable: &str, value: &str) -> Self {
        if !self.variables.contains_key(variable) {
            self.variables
                .insert(variable.to_string(), value.to_string());
        } else {
            log::warn!(
                "Overriding an already set variable {variable}, please consider using `with_overwrite` "
            );
        }
        self
    }

    /// Adds a variable to the prompt, overwriting it if it already exists.
    fn with_overwrite(mut self, variable: &str, value: &str) -> Self {
        self.variables
            .insert(variable.to_string(), value.to_string());
        self
    }

    /// Sets the file extension for the prompt template.
    fn with_extension(mut self, extension: &str) -> Self {
        self.extension = extension.to_string();
        self
    }

    /// Returns the file path of the prompt template.
    fn file_path(&self) -> PathBuf {
        let file_name = format!("{}.{}", self.name, self.extension);
        self.env.prompt_dir.join(file_name)
    }

    /// Returns the name of the prompt.
    fn name(&self) -> &str {
        &self.name
    }

    /// Returns the file extension of the prompt.
    fn extension(&self) -> &str {
        &self.extension
    }

    /// Returns a reference to the prompt's variables.
    fn variables(&self) -> &PromptVariables {
        &self.variables
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_new_prompt() {
        let prompt = Prompt::new("test_prompt");
        assert_eq!(prompt.name(), "test_prompt");
        assert_eq!(prompt.extension(), "prompt");
        assert!(prompt.variables().is_empty());
    }

    #[test]
    fn test_with() {
        let prompt = Prompt::new("test_prompt").with("key", "value");
        assert_eq!(prompt.variables().get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_with_does_not_overwrite() {
        let prompt = Prompt::new("test_prompt")
            .with("key", "value1")
            .with("key", "value2");
        assert_eq!(prompt.variables().get("key"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_with_overwrite() {
        let prompt = Prompt::new("test_prompt")
            .with_overwrite("key", "value1")
            .with_overwrite("key", "value2");
        assert_eq!(prompt.variables().get("key"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_with_extension() {
        let prompt = Prompt::new("test_prompt").with_extension("txt");
        assert_eq!(prompt.extension(), "txt");
    }

    #[test]
    fn test_file_path() {
        // Unset PROMPT_DIR to test default
        unsafe {
            env::remove_var("PROMPT_DIR");
        }
        let prompt = Prompt::new("system/test").with_extension("md");
        let path = prompt.file_path();
        assert!(path.ends_with("prompts/system/test.md"));
    }

    #[test]
    fn test_file_path_with_custom_prompt_dir() {
        let tmp_dir = tempdir().unwrap();
        let custom_dir = tmp_dir.path().to_str().unwrap();
        unsafe {
            env::set_var("PROMPT_DIR", custom_dir);
        }
        let prompt = Prompt::new("user/test");
        let path = prompt.file_path();
        assert_eq!(path, PathBuf::from(custom_dir).join("user/test.prompt"));
        unsafe {
            env::remove_var("PROMPT_DIR");
        }
    }

    #[test]
    fn test_generate_prompt_with_custom_env() {
        let tmp_dir = tempdir().unwrap();
        let custom_dir = tmp_dir.path();
        let template_path = custom_dir.join("test_template.txt");
        fs::write(&template_path, "Hello, {{ name }}!").unwrap();

        let env = PromptEnvironment::from_directory(custom_dir.to_str().unwrap());
        let prompt = Prompt::new_with_env("test_template", env)
            .with_extension("txt")
            .with("name", "World");

        let generated_string = prompt.generate();
        assert_eq!(generated_string, "Hello, World!");
    }

    #[test]
    fn test_base_prompt_default() {
        let prompt = Prompt::new("system/base");
        let result = prompt.generate();

        assert_eq!(
            "You are a helpful AI assistant. Your role is to assist the user.\n",
            result
        )
    }
}
