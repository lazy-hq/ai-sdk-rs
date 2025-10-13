use crate::error::{Error, Result};
use derive_builder::Builder;
use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;

pub type ToolFn = Box<dyn Fn(Value) -> std::result::Result<String, String> + Send + Sync>;

#[derive(Clone)]
pub struct ToolExecute {
    inner: Arc<ToolFn>,
}

impl ToolExecute {
    pub fn call(&self, map: Value) -> Result<String> {
        (*self.inner)(map).map_err(Error::ToolCallError)
    }

    pub fn new(f: ToolFn) -> Self {
        Self { inner: Arc::new(f) }
    }
}

impl Default for ToolExecute {
    fn default() -> Self {
        Self::new(Box::new(|_| Ok("".to_string())))
    }
}

impl Serialize for ToolExecute {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str("ToolExecuteCall")
    }
}

impl<'de> Deserialize<'de> for ToolExecute {
    fn deserialize<D>(_: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::default())
    }
}

#[derive(Builder, Clone, Default)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct Tool {
    /// The name of the tool
    pub name: String,
    /// AI friendly description
    pub description: String,
    /// The input schema of the tool as json schema
    pub input_schema: Schema,
    /// The output schema of the tool. AI will use this to generate outputs.
    pub execute: ToolExecute,
}

impl Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

impl Tool {
    pub fn new() -> Self {
        Self {
            name: "".to_string(),
            description: "".to_string(),
            input_schema: Schema::default(),
            execute: ToolExecute::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolList {
    pub tools: Arc<Mutex<Vec<Tool>>>,
}

impl ToolList {
    pub fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools: Arc::new(Mutex::new(tools)),
        }
    }

    pub fn add_tool(&mut self, tool: Tool) {
        self.tools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(tool);
    }

    pub async fn execute(&self, tool_info: ToolCallInfo) -> JoinHandle<Result<String>> {
        let tools = self.tools.clone();
        tokio::spawn(async move {
            let tools = tools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let tool = tools.iter().find(|tool| tool.name == tool_info.tool.name);

            match tool {
                Some(tool) => tool.execute.call(tool_info.input),
                None => Err(crate::error::Error::ToolCallError(
                    "Tool not found".to_string(),
                )),
            }
        })
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
/// Describes a tool
pub struct ToolDetails {
    // the name of the tool, usually a function name.
    pub name: String,
    // uniquely identifies a tool, provided by the LLM.
    pub id: String,
}

/// Contains information necessary to call a tool
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ToolCallInfo {
    pub tool: ToolDetails,
    pub input: serde_json::Value,
}

impl ToolCallInfo {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            tool: ToolDetails {
                name: name.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn name(&mut self, name: impl Into<String>) {
        self.tool.name = name.into();
    }

    pub fn id(&mut self, id: impl Into<String>) {
        self.tool.id = id.into();
    }

    pub fn input(&mut self, inp: serde_json::Value) {
        self.input = inp;
    }
}

/// Contains information from a tool
#[derive(Debug, Clone)]
pub struct ToolResultInfo {
    pub tool: ToolDetails,
    pub output: Result<serde_json::Value>,
}

impl Default for ToolResultInfo {
    fn default() -> Self {
        Self {
            tool: ToolDetails::default(),
            output: Ok(serde_json::Value::Null),
        }
    }
}

impl ToolResultInfo {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            tool: ToolDetails {
                name: name.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn name(&mut self, name: impl Into<String>) {
        self.tool.name = name.into();
    }

    pub fn id(&mut self, id: impl Into<String>) {
        self.tool.id = id.into();
    }

    pub fn output(&mut self, inp: serde_json::Value) {
        self.output = Ok(inp);
    }
}

// tests
#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use super::*;
    use aisdk_macros::tool;
    use std::collections::HashMap;

    #[tool]
    /// This is The Description of an example tool.
    pub fn my_example_tool(a: u8, b: Option<u8>) -> Tool {
        Ok(format!("{}{}", a, b.unwrap_or(0)))
    }

    #[tokio::test]
    async fn test_tool_macro_with_no_args() {
        schemars::schema_for!(String);
        let tool = my_example_tool();

        assert_eq!(tool.name, "my_example_tool");
        assert_eq!(
            tool.description,
            " This is The Description of an example tool."
        );
        let schema_properties = tool
            .input_schema
            .as_object()
            .unwrap()
            .get("properties")
            .unwrap();
        assert_eq!(
            schema_properties.get("a").unwrap().get("format").unwrap(),
            &serde_json::Value::String("uint8".to_string())
        );
        assert_eq!(
            schema_properties.get("b").unwrap().get("format").unwrap(),
            &serde_json::Value::String("uint8".to_string())
        );
        assert_eq!(
            schema_properties.get("b").unwrap().get("type").unwrap(),
            &serde_json::Value::Array(vec![
                serde_json::Value::String("integer".to_string()),
                serde_json::Value::String("null".to_string())
            ])
        );
        assert_eq!(
            tool.execute
                .call(Value::Object(
                    HashMap::from([
                        ("a".to_string(), 1.into()),
                        ("b".to_string(), Option::<u32>::None.into())
                    ])
                    .into_iter()
                    .collect()
                ))
                .unwrap(),
            "10".to_string()
        );
    }

    #[tool(name = "the-name-for-this-tool")]
    pub fn my_example_tool_with_name(name: String, a: u8, b: Option<u8>) -> Tool {
        Ok(format!("{}{}", a, b.unwrap_or(0)))
    }

    #[test]
    fn test_tool_macro_with_name() {
        let tool = my_example_tool_with_name();
        assert!(tool.name != "my-example-tool-with-name");
        assert_eq!(tool.name, "the-name-for-this-tool");
    }

    #[tool(desc = "the-description-for-this-tool")]
    /// This is The Description of an example tool.
    pub fn my_example_tool_with_description(name: String, a: u8, b: Option<u8>) -> Tool {
        Ok(format!("{}{}", a, b.unwrap_or(0)))
    }

    #[test]
    /// This is The Description of an example tool.
    fn test_tool_macro_with_description() {
        let tool = my_example_tool_with_description();
        assert!(tool.description != " This is The Description of an example tool.");
        assert_eq!(tool.description, "the-description-for-this-tool");
    }

    #[tool(
        name = "the-name-for-this-tool",
        desc = "the-description-for-this-tool"
    )]
    /// This is The Description of an example tool.
    pub fn my_example_tool_with_name_and_description(name: String, a: u8, b: Option<u8>) -> Tool {
        Ok(format!("{}{}", a, b.unwrap_or(0)))
    }

    #[test]
    fn test_tool_macro_with_name_and_description() {
        let tool = my_example_tool_with_name_and_description();
        assert!(tool.name != "my-example-tool-with-name-and-description");
        assert_eq!(tool.name, "the-name-for-this-tool");
        assert!(tool.description != " This is The Description of an example tool.");
        assert_eq!(tool.description, "the-description-for-this-tool");
    }

    #[tool(
        desc = "the-description-for-this-tool",
        name = "the-name-for-this-tool"
    )]
    /// This is The Description of an example tool.
    pub fn my_example_tool_with_description_and_name(name: String, a: u8, b: Option<u8>) -> Tool {
        Ok(format!("{}{}", a, b.unwrap_or(0)))
    }

    #[test]
    fn test_tool_macro_with_description_and_name() {
        let tool = my_example_tool_with_description_and_name();
        assert!(tool.name != "my-example-tool-with-description-and-name");
        assert_eq!(tool.name, "the-name-for-this-tool");
        assert!(tool.description != " This is The Description of an example tool.");
        assert_eq!(tool.description, "the-description-for-this-tool");
    }

    #[test]
    fn test_argument_json_schema() {}
}
