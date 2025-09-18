use crate::error::{Error, Result};
use derive_builder::Builder;
use schemars::Schema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

pub type ToolFn =
    Box<dyn FnMut(HashMap<String, Value>) -> std::result::Result<String, String> + Send + Sync>;

#[derive(Clone)]
pub struct ToolExecute {
    inner: Arc<Mutex<ToolFn>>,
}

impl ToolExecute {
    pub fn call(&self, map: HashMap<String, Value>) -> Result<String> {
        let mut guard = self.inner.lock().unwrap();
        (guard)(map).map_err(|e| {
            Error::Other(e) // TODO: use tool specific errors
        })
    }

    pub fn new(f: ToolFn) -> Self {
        Self {
            inner: Arc::new(Mutex::new(f)),
        }
    }
}

impl Default for ToolExecute {
    fn default() -> Self {
        Self::new(Box::new(|_| Ok("".to_string())))
    }
}

#[derive(Builder, Clone, Serialize, Deserialize, Default)]
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

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use aisdk_macros::tool;

    #[tool]
    /// This is The Description of an example tool.
    pub fn my_example_tool(a: u8, b: Option<u8>) -> Tool {
        Ok(format!("{}{}", a, b.unwrap_or(0)))
    }

    #[test]
    fn test_macro() {
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
                .call(HashMap::from([
                    ("a".to_string(), 1.into()),
                    ("b".to_string(), Option::<u32>::None.into())
                ]))
                .unwrap(),
            "10".to_string()
        );
    }

    #[test]
    fn test_argument_json_schema() {}
}
