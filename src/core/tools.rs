use crate::error::Result;
use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use std::any::Any;

pub struct Tool<Inputs>
where
    Inputs: JsonSchema + Serialize + IntoIterator,
{
    //pub type ToolInput =
    /// The name of the tool
    pub name: String,
    /// AI friendly description
    pub description: String,
    /// The output schema of the tool. AI will use this to generate outputs.
    pub execute: fn(Inputs) -> Result<String>,
}

impl<Inputs> Tool<Inputs>
where
    Inputs: JsonSchema + Serialize + IntoIterator,
{
    pub fn new(name: &str, description: &str, execute: fn(Inputs) -> Result<String>) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            execute,
        }
    }

    /// Generate JSON schema for the input type
    pub fn argument_schema(&self) -> schemars::Schema {
        schema_for!(Inputs)
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argument_json_schema() {
        type ToolInput = Vec<Option<String>>;

        // This works too
        //let tool = Tool::new("test", "test", |input: ToolInput| Ok(format!("{:?}", input)));
        let tool = Tool::<ToolInput>::new("test", "test", |input: ToolInput| {
            Ok(format!("{:?}", input))
        });

        let schema = tool.argument_schema();
        eprintln!("{}", schema.clone().to_value());
        assert_eq!(schema_for!(ToolInput), schema);
    }
}
