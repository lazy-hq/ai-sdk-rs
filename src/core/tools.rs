use crate::error::{Error, Result};
use aisdk_macros::tool_factory;
use schemars::{JsonSchema, Schema, schema_for};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

pub struct Tool {
    /// The name of the tool
    pub name: String,
    /// AI friendly description
    pub description: String,
    /// Tool inputs
    pub input_schema: Schema,
    /// The output schema of the tool. AI will use this to generate outputs.
    pub execute: Box<dyn FnMut(HashMap<String, Value>) -> Result<String>>,
}

impl Tool {
    pub fn new() -> Self {
        Self {
            name: "".to_string(),
            description: "".to_string(),
            input_schema: Schema::default(),
            execute: Box::new(|_| Ok("".to_string())),
        }
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[tool_factory]
    /// This is The Description of an example tool.
    pub fn my_example_tool(a: u8, b: Option<u8>) -> Tool {
        format!("{}{}", a, b.unwrap_or(0))
    }

    #[test]
    fn test_macro() {
        schemars::schema_for!(String);
        let mut tool = my_example_tool();

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
            (tool.execute)(HashMap::from([
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
