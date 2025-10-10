use proc_macro::TokenStream;
use quote::quote;
use syn::parse::Parser;
use syn::{
    parse_macro_input, punctuated::Punctuated, Expr, ExprLit, FnArg, ItemFn, Lit, Meta,
    MetaNameValue, Pat, Token,
};

#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;
    let block = &input_fn.block;
    let inputs = &input_fn.sig.inputs;
    let attrs = &input_fn.attrs;
    let args_parser = Punctuated::<MetaNameValue, Token![,]>::parse_terminated;
    let args = args_parser.parse(_attr);

    let (name_arg, description_arg) = if let Ok(args) = args {
        let mut name: Option<String> = None;
        let mut description: Option<String> = None;

        for arg in args {
            if arg.path.is_ident("desc") {
                if let Expr::Lit(lit) = &arg.value {
                    if let Lit::Str(str_lit) = &lit.lit {
                        description = Some(str_lit.value());
                    }
                }
            } else if arg.path.is_ident("name") {
                if let Expr::Lit(lit) = &arg.value {
                    if let Lit::Str(str_lit) = &lit.lit {
                        name = Some(str_lit.value());
                    }
                }
            }
        }

        (name, description)
    } else {
        (None, None)
    };

    let description = if let Some(desc) = description_arg {
        desc
    } else {
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

        doc_comments.join("\n")
    };

    let name = if let Some(name) = name_arg {
        name
    } else {
        fn_name.to_string()
    };

    let binding_tokens: Vec<_> = inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    let ident = &pat_ident.ident;
                    let ty = &*pat_type.ty;
                    let ident_str = ident.to_string();
                    Some(quote! {
                        let #ident: #ty = serde_json::from_value(
                            inp.as_object().unwrap().get(#ident_str).unwrap().clone()
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
        })
        .collect();

    // Generate the struct definition
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

    let expanded = quote! {
        // TODO: If there is a way to remove the usage of Tool and ToolExecute here it would be
        // nice. currently the user has to import these types.
        #[allow(unused_variables)]
        #vis fn #fn_name() -> Tool {
            // TODO: There is a possibiltiy to run schema genration during compile time here.
            // This will potentiolly remove the need for additional runtime dependency.
            use schemars::{schema_for, JsonSchema, Schema};
            use serde::Serialize;
            use std::collections::HashMap;

            #[derive(JsonSchema, Serialize, Debug)]
            //#[schemars(deny_unknown_fields)]
            struct Function {
                // Please add struct fields here
                #(#struct_fields),*
            }

            let input_schema = schema_for!(Function);
            // End

            let mut tool = Tool::new();

            tool.name = #name.to_string();
            tool.description = #description.to_string();
            tool.input_schema = input_schema;
            tool.execute = ToolExecute::new(Box::new(|inp| -> std::result::Result<String, String> {
                // TODO: Do `input_schema` validation on inp
                // Extract all parameters from the HashMap here
                #(#binding_tokens)*
                #block
            }));

            tool
        }
    };

    TokenStream::from(expanded)
}
