//! Example demonstrating OpenAI provider integration with models.dev.
//!
//! This example shows how to use the OpenAI provider with the ModelsDevAware trait
//! to automatically create providers from models.dev information.

#[cfg(all(feature = "models-dev", feature = "openai"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aisdk::models_dev::traits::ModelsDevAware;
    use aisdk::models_dev::types::{
        ApiInfo, DocInfo, EnvVar, Modalities, Model, ModelCost, ModelLimit, NpmInfo, Provider,
    };
    use std::collections::HashMap;

    // Create a mock OpenAI provider info (normally this would come from models.dev API)
    let openai_provider_info = Provider {
        id: "openai".to_string(),
        name: "OpenAI".to_string(),
        npm: NpmInfo {
            name: "@ai-sdk/openai".to_string(),
            version: "1.0.0".to_string(),
        },
        env: vec![EnvVar {
            name: "OPENAI_API_KEY".to_string(),
            description: "OpenAI API key".to_string(),
            required: true,
        }],
        doc: DocInfo {
            url: "https://platform.openai.com/docs".to_string(),
            metadata: HashMap::new(),
        },
        api: ApiInfo {
            base_url: "https://api.openai.com/v1".to_string(),
            version: None,
            config: HashMap::new(),
        },
        models: vec![Model {
            id: "gpt-4o".to_string(),
            name: "GPT-4o".to_string(),
            description: "Latest GPT-4 model".to_string(),
            cost: ModelCost::default(),
            limits: ModelLimit {
                context: 128000,
                output: 4096,
                metadata: HashMap::new(),
            },
            modalities: Modalities {
                input: vec!["text".to_string()],
                output: vec!["text".to_string()],
            },
            metadata: HashMap::new(),
        }],
    };

    // Check if OPENAI_API_KEY is set
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("⚠️  OPENAI_API_KEY environment variable is not set.");
        println!("Please set it to run this example:");
        println!("export OPENAI_API_KEY=your-api-key");
        return Ok(());
    }

    // Create OpenAI provider from models.dev info
    let openai =
        aisdk::providers::openai::OpenAI::from_models_dev_info(&openai_provider_info, None)
            .expect("Failed to create OpenAI provider from models.dev info");

    println!("✅ Successfully created OpenAI provider from models.dev info!");
    println!("   Provider name: {}", openai.settings.provider_name);
    println!("   Model: {}", openai.settings.model_name);
    println!("   Base URL: https://api.openai.com/v1");

    // Test the provider with a simple generation
    use aisdk::core::language_model::LanguageModel;
    use aisdk::core::types::LanguageModelCallOptions;

    let mut openai_provider = openai;

    println!("\n🔄 Testing provider with a simple generation...");

    let options = LanguageModelCallOptions::builder()
        .messages(vec![aisdk::core::types::Message::User(
            "Hello! Please respond with just 'Hello from OpenAI!'".into(),
        )])
        .build()
        .expect("Failed to build options");

    match openai_provider.generate(options).await {
        Ok(response) => {
            println!("✅ Generation successful!");
            println!("   Response: {}", response.text);
            println!("   Model: {:?}", response.model);
        }
        Err(e) => {
            println!("❌ Generation failed: {}", e);
        }
    }

    Ok(())
}

#[cfg(not(all(feature = "models-dev", feature = "openai")))]
fn main() {
    println!("⚠️  This example requires both 'models-dev' and 'openai' features to be enabled.");
    println!(
        "Run with: cargo run --example openai_models_dev_integration --features='openai models-dev'"
    );
}
