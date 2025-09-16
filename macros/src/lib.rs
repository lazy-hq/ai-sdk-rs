use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprLit, FnArg, ItemFn, Lit, Meta, Pat, ReturnType};
use schemars::{JsonSchema, Schema, schema_for};
use serde::{Deserialize, Serialize};

#[proc_macro_attribute]
pub fn tool_factory(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;
    let block = &input_fn.block;
    let inputs = &input_fn.sig.inputs;
    let attrs = &input_fn.attrs;

    // Extract doc comments
    let doc_comments: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let Meta::NameValue(meta_name_value) = &attr.meta {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(lit_str),
                        ..
                    }) = &meta_name_value.value
                    {
                        let doc = lit_str.value();
                        Some(doc)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    let description = doc_comments.join("\n");

    let struct_fields = inputs.iter().filter_map(|arg| {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let ident = &pat_ident.ident;
                let ty = &*pat_type.ty;
                Some(quote! { #ident: #ty })
            } else {
                None
            }
        } else {
            None
        }
    });

    let binding_tokens: Vec<_> = inputs.iter().filter_map(|arg| {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let ident = &pat_ident.ident;
                let ty = &*pat_type.ty;
                let ident_str = ident.to_string();
                Some(quote! {
                    let #ident: #ty = serde_json::from_value(
                        inp.remove(#ident_str)
                            .unwrap_or_else(|| todo!("Missing required parameter: {}", #ident_str))
                    ).unwrap_or_else(|e| {
                        todo!("Failed to deserialize {}: {}", #ident_str, e)
                    });
                })
            } else {
                None
            }
        } else {
            None
        }
    }).collect();

    let expanded = quote! {
        #vis fn #fn_name() -> Tool {
            let mut tool = Tool::new();

            #[derive(JsonSchema, Serialize, Debug)]
            struct ToolInput {
                #(#struct_fields),*
            }

            tool.name = stringify!(#fn_name).to_string();  // TODO: Change to better formatted text
            // than snake case
            tool.description = #description.to_string();
            tool.input_schema = schema_for!(ToolInput);
            tool.execute = ToolExecute::new(Box::new(|mut inp: HashMap<String, serde_json::Value>| -> Result<String> {
                // TODO: Do `input_schema` validation on inp
                // Extract all parameters from the HashMap here
                #(#binding_tokens)*
                if !inp.is_empty() {
                    return Err(Error::Other(format!("Unexpected parameters: {:?}", inp.keys().collect::<Vec<_>>())));
                }
                Ok(#block)
                //Ok(format!("{:?}", result))
            }));
            tool
        }
    };

    TokenStream::from(expanded)
}
