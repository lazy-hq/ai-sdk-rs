use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprLit, ItemFn, Lit, Meta};

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

    let expanded = quote! {
        #vis fn #fn_name() -> Tool {
            let mut tool = Tool::new(); 
            tool.name = stringify!(#fn_name).to_string();
            tool.description = #description.to_string();
            tool
            //Tool {
            //    name: stringify!(#fn_name).to_string(),
            //    description: #description.to_string(),
            //    input_schema: schemars::schema_for!(String),
            //    execute: Box::new(|inp| -> Result<String> { Ok("place holder".to_string()) }),
            //}
        }
    };

    TokenStream::from(expanded)
}
